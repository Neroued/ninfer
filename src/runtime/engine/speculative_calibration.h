#pragma once

#include "core/startup.h"
#include "models/common/speculative/calibration.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/request.h"
#include "runtime/executor/executor.h"

#include <algorithm>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>

namespace ninfer::runtime {

struct SpeculativeCalibrationResult {
    std::uint32_t anchors;
    std::uint32_t contexts;
    double seconds;
};

namespace detail {

template <class Instance>
class SpeculativeCalibration {
    using Model        = typename Instance::ModelContract;
    using Sequence     = typename Model::SequenceHandle;
    using Checkpoint   = typename Model::CheckpointHandle;
    using PendingBatch = typename Model::PendingBatch;
    using Clock        = std::chrono::steady_clock;
    using Probe        = models::speculative::CalibrationProbeSpec;
    using Route        = models::speculative::RoundRoute;

public:
    explicit SpeculativeCalibration(Instance& instance)
        : instance_(instance), session_(*instance.session), executor_(*instance.executor),
          constraints_(session_.calibration_constraints()) {}

    std::optional<SpeculativeCalibrationResult> run(const StartupObserver& observer,
                                                    std::string_view prompt_text) {
        const auto schedule = models::speculative::build_corner_calibration_schedule(constraints_);
        if (schedule.empty()) return {};
        started_ = Clock::now();
        StartupPhaseScope phase(observer, StartupPhase::SpeculativeCalibration);
        const auto pattern = instance_.frontend.tokenize_text(prompt_text);
        if (pattern.empty()) throw std::logic_error("speculative calibration prompt is empty");
        try {
            run_context(schedule.front().context_tokens, pattern, schedule);
            const auto long_probe = std::ranges::find_if(schedule, [&](const Probe& probe) {
                return probe.context_tokens != schedule.front().context_tokens;
            });
            if (long_probe != schedule.end() && elapsed_ms() + 800.0 < constraints_.budget_ms)
                run_context(long_probe->context_tokens, pattern, schedule);
            session_.fit_speculative_cost_model(anchors_, elapsed_ms());
            executor_.reset_startup_observations();
        } catch (...) {
            session_.fail_all_cleanup();
            throw;
        }
        const SpeculativeCalibrationResult result{static_cast<std::uint32_t>(anchors_.size()),
                                                  contexts_, elapsed_ms() * 1e-3};
        phase.complete();
        return result;
    }

private:
    [[nodiscard]] double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - started_).count();
    }

    [[nodiscard]] bool in_budget() const { return elapsed_ms() < constraints_.budget_ms; }

    template <class Event>
    Event wait(std::optional<ExecutionTicket> ticket = {}) {
        for (;;) {
            if (auto event = session_.poll()) {
                if ((!ticket || event->ticket == *ticket) &&
                    std::holds_alternative<Event>(event->result))
                    return std::get<Event>(std::move(event->result));
                continue;
            }
            if (!executor_.has_pending_work() && !session_.has_context_operation())
                throw std::logic_error("speculative calibration execution ended without its event");
            executor_.flush();
            std::this_thread::yield();
        }
    }

    void drain() {
        while (session_.poll() || executor_.has_pending_work() ||
               session_.has_context_operation()) {
            executor_.flush();
            std::this_thread::yield();
        }
    }

    void commit(PendingBatch&& pending, bool terminal = false) {
        std::vector<CommitDecision> decisions;
        decisions.reserve(pending.row_count());
        for (const auto count : pending.row_counts())
            decisions.push_back(
                {.accepted_tokens = static_cast<std::uint32_t>(count), .terminal = terminal});
        (void)session_.commit(std::move(pending), decisions);
    }

    std::optional<Sequence> bind(const typename Model::RequestBasePlan& plan, std::uint32_t lane,
                                 PlacementId placement, std::optional<Checkpoint> source) {
        const auto candidate = session_.inspect_source(plan, source, false, {}, {}, placement);
        if (!candidate || !session_.start_binding(plan, {lane}, *candidate)) return {};
        for (;;) {
            auto progress = wait<typename Model::ContextProgress>();
            if (progress.kind == decltype(progress.kind)::Bind && progress.complete)
                return progress.sequence;
        }
    }

    bool prefill(Sequence sequence, bool terminal) {
        const typename Model::ExecutionUnit unit{sequence, Model::ExecutionUnitKind::Prefill, 0};
        while (in_budget()) {
            if (!session_.reserve_units({&unit, 1})) return false;
            const auto ticket = session_.start_prefill(sequence);
            if (!ticket) return false;
            auto result = wait<typename Model::PrefillProgress>(*ticket);
            if (result.rows.size() != 1)
                throw std::logic_error("speculative calibration prefill changed membership");
            if (result.rows.front().capture_ready) session_.skip_capture(sequence);
            if (result.pending) commit(std::move(*result.pending), terminal);
            drain();
            if (result.pending) return true;
        }
        return false;
    }

    bool launch(std::span<const Sequence> members, const Route& route,
                std::span<const std::uint32_t> extents, bool timed) {
        if (!in_budget()) return false;
        std::vector<typename Model::ExecutionUnit> units;
        units.reserve(members.size());
        for (const auto member : members) {
            if (std::uint64_t(session_.decode_frontier(member)) + constraints_.model.max_drafts +
                    1 >
                constraints_.max_context)
                return false;
            units.push_back({member, Model::ExecutionUnitKind::Decode, 1});
        }
        if (!session_.reserve_units(units)) return false;
        const auto ticket = session_.start_decode_probe(members, route, extents);
        if (!ticket) {
            session_.release_units(members);
            return false;
        }
        const auto query = session_.decode_cost_query(*ticket);
        if (!query) throw std::logic_error("speculative calibration probe has no cost geometry");
        commit(wait<PendingBatch>(*ticket));
        const auto event = wait<ExecutionComplete>(*ticket);
        if (timed) anchors_.push_back({*query, double(event.sample.latency.total_ns) * 1.0e-6});
        return true;
    }

    void release(Sequence member) {
        if (session_.abort(member).status != ConsumeStatus::Consumed)
            throw std::logic_error("speculative calibration sequence did not release");
    }

    Route ordinary_route(std::uint32_t rows) const {
        const auto owners = constraints_.model.owner_count;
        const auto local  = (rows + owners - 1) / owners;
        if (constraints_.path == ExecutionPath::Eager)
            return Route::eager(constraints_.batches->exact(rows, local),
                                constraints_.model.backend, constraints_.model.max_drafts, 0);
        return constraints_.routes->select_route(constraints_.batches->decode(rows, local),
                                                 constraints_.model.backend, 0);
    }

    void run_context(std::uint32_t length, std::span<const TokenId> pattern,
                     std::span<const Probe> schedule) {
        std::uint32_t max_rows = 0;
        for (const auto& probe : schedule)
            if (probe.context_tokens == length) max_rows = std::max(max_rows, probe.logical_rows);
        std::vector<TokenId> ids(length);
        for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = pattern[i % pattern.size()];

        ResolvedExecutionOptions options;
        options.sampling.temperature    = 1.0f;
        options.sampling.top_k          = 20;
        options.requested_output_tokens = 4096;
        const auto placements           = session_.placement_descriptions();
        std::vector<std::optional<Checkpoint>> sources(placements.size());
        const auto make_plan = [&] {
            return session_.plan_request(
                instance_.frontend.prepare_tokens(ids, options.allow_prefix_reuse), options);
        };
        options.allow_prefix_reuse = max_rows > 1;
        if (options.allow_prefix_reuse) {
            options.sampling.seed = 999;
            auto root             = make_plan();
            if (root.summary().publish_continuation) {
                for (std::size_t owner = 0;
                     owner < std::min<std::size_t>(max_rows, placements.size()) && in_budget();
                     ++owner) {
                    const auto sequence = bind(root, 0, {placements[owner].id}, {});
                    if (!sequence) break;
                    if (prefill(*sequence, true)) {
                        const auto finished = session_.finish(*sequence);
                        if (finished.status != ConsumeStatus::Consumed)
                            throw std::logic_error("speculative calibration root did not finish");
                        sources[owner] = finished.checkpoint;
                        if (sources[owner] &&
                            session_.checkpoint_summary(*sources[owner]).frontier != length)
                            throw std::logic_error(
                                "calibration checkpoint changed the prompt frontier");
                    } else {
                        release(*sequence);
                    }
                    drain();
                }
            }
        }

        std::vector<Sequence> members;
        members.reserve(max_rows);
        for (std::uint32_t b = 0; b < max_rows && in_budget(); ++b) {
            const auto owner           = b % placements.size();
            options.allow_prefix_reuse = sources[owner].has_value();
            options.sampling.seed      = 1000 + b;
            const auto plan            = make_plan();
            const auto sequence        = bind(plan, b, {placements[owner].id}, sources[owner]);
            if (!sequence) break;
            if (!prefill(*sequence, false)) {
                release(*sequence);
                drain();
                break;
            }
            members.push_back(*sequence);
        }
        if (!members.empty() && launch(members, ordinary_route(members.size()),
                                       std::vector<std::uint32_t>(members.size()), false)) {
            ++contexts_;
            for (const auto& probe : schedule) {
                if (!in_budget()) break;
                if (probe.context_tokens != length || probe.logical_rows > members.size()) continue;
                const auto rows = std::span<const Sequence>(members).first(probe.logical_rows);
                bool ready      = true;
                for (std::uint32_t i = 0; i < probe.warmup_launches && ready; ++i)
                    ready = launch(rows, probe.route, probe.verify_drafts, false);
                for (std::uint32_t i = 0; i < probe.timed_launches && ready; ++i) {
                    if (i && elapsed_ms() > 0.7 * constraints_.budget_ms) break;
                    ready = launch(rows, probe.route, probe.verify_drafts, true);
                }
                if (!ready) break;
            }
        }
        for (const auto member : members) release(member);
        for (const auto source : sources)
            if (source && !session_.release_checkpoint(*source))
                throw std::logic_error("speculative calibration checkpoint did not release");
        drain();
    }

    Clock::time_point started_;
    Instance& instance_;
    typename Model::Session& session_;
    Executor<typename Model::ExecutionTypes>& executor_;
    models::speculative::CalibrationConstraints constraints_;
    std::vector<models::speculative::TimedCostAnchor> anchors_;
    std::uint32_t contexts_ = 0;
};

} // namespace detail

// Called on the executor's bound thread before the Session admits user requests.
template <class Instance>
std::optional<SpeculativeCalibrationResult> calibrate_speculative(
    Instance& instance, const StartupObserver& observer,
    std::string_view prompt_text =
        "A field notebook records the date, weather, and observations from each walk. ") {
    return detail::SpeculativeCalibration<Instance>(instance).run(observer, prompt_text);
}

} // namespace ninfer::runtime
