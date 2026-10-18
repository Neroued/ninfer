#pragma once

#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/session/internal.h"
#include "runtime/executor/local_executor.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <map>
#include <stdexcept>
#include <thread>
#include <functional>
#include <memory>
#include <utility>

namespace ninfer::test::qwen3_5 {
namespace q35 = ninfer::models::qwen3_5;

// Tests observe the native implementation at the same submission and event boundaries used by
// RankProgram. No observer or synchronous execution entry is added to the product interface.
class ObservedRank {
public:
    ObservedRank(const q35::execution::Parameters& parameters,
                 const q35::detail::SequencePlanImpl& plan, DeviceContext& device)
        : implementation(parameters, plan, device, {}) {
        implementation.initialize_graphs();
    }

    [[nodiscard]] runtime::ExecutionCapacity capacity() const noexcept { return {1, 1}; }

    void validate_work(const q35::Work& work, runtime::FrameKind kind) const {
        implementation.validate_work(work, kind);
    }

    void validate_decision(runtime::ExecutionTicket ticket,
                           const runtime::DecisionInput& input) const {
        implementation.validate_decision(ticket, input);
    }

    void prepare_commit(runtime::ExecutionTicket ticket, q35::Commit& commit) const {
        implementation.prepare_commit(ticket, commit);
    }

    void start(runtime::ExecutionRequest<q35::ExecutionTypes>&& request) {
        if (submitted) submitted(request);
        implementation.start(std::move(request));
    }

    void supply(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                runtime::DecisionInput&& input) {
        implementation.supply(ticket, command, std::move(input));
    }

    void commit(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                q35::Commit&& commit) {
        auto description = committed ? std::optional(commit) : std::nullopt;
        implementation.commit(ticket, command, std::move(commit));
        if (committed) committed(ticket, *description);
    }

    void discard(runtime::ExecutionTicket ticket, runtime::CommandSequence command) {
        implementation.discard(ticket, command);
    }

    [[nodiscard]] std::optional<runtime::ExecutionEvent<q35::ExecutionTypes>> poll() {
        auto event = implementation.poll();
        if (event && observed) observed(*event);
        return event;
    }

    void drain() { implementation.drain(); }

    [[nodiscard]] MemorySummary memory_summary() const { return implementation.memory_summary(); }

    void reset_memory_peaks() noexcept { implementation.reset_memory_peaks(); }

    q35::detail::RankProgramImpl implementation;
    std::function<void(const runtime::ExecutionRequest<q35::ExecutionTypes>&)> submitted;
    std::function<void(const runtime::ExecutionEvent<q35::ExecutionTypes>&)> observed;
    std::function<void(runtime::ExecutionTicket, const q35::Commit&)> committed;
};

// Reuses immutable loaded weights across test geometries. Each worker still owns its complete
// native Parameters, RankProgram, DeviceContext and execution resources.
struct SharedModel {
    std::shared_ptr<const q35::ModelShard> value;

    operator const q35::ModelShard&() const noexcept { return *value; }
};

struct ObservedTypes {
    using ModelShard     = SharedModel;
    using Parameters     = q35::execution::Parameters;
    using Profile        = q35::execution::ModelProfile;
    using RankProgram    = ObservedRank;
    using ExecutionTypes = q35::ExecutionTypes;
};

class NativeRankFixture {
public:
    NativeRankFixture(std::shared_ptr<const q35::ModelDefinition> definition,
                      std::shared_ptr<const q35::ModelShard> model,
                      std::shared_ptr<const q35::execution::ModelProfile> profile,
                      const EngineOptions& options, std::uint32_t page_groups)
        : definition(std::move(definition)) {
        auto device   = std::make_unique<DeviceContext>();
        device_       = device.get();
        auto instance = std::make_unique<runtime::RankModelInstance<ObservedTypes>>(
            std::make_unique<SharedModel>(SharedModel{std::move(model)}), profile);
        auto planner        = q35::make_sequence_planner(profile, *device, options);
        auto plan           = std::move(planner).finalize(page_groups);
        session_plan        = q35::plan_session(plan);
        auto native_planner = q35::detail::make_sequence_planner_impl(profile, *device, options);
        auto native_plan =
            q35::detail::finalize_sequence_plan_impl(std::move(native_planner), page_groups);
        instance->program =
            std::make_unique<ObservedRank>(instance->parameters, *native_plan, *device);
        observed_             = instance->program.get();
        const auto& resources = observed_->implementation.resources;
        session_plan.context.pools[0].transfer =
            resources.decoder.text_kv.pages().transfer_profile();
        if (const auto* backend = resources.backend_kv())
            session_plan.context.pools[1].transfer = backend->pages().transfer_profile();
        auto worker = std::make_unique<runtime::RankWorker<ObservedTypes>>(std::move(device),
                                                                           std::move(instance));
        executor_   = std::make_unique<runtime::LocalExecutor<ObservedTypes>>(
            std::move(worker), runtime::SessionEpoch{next_epoch_.fetch_add(1)});
    }

    [[nodiscard]] DeviceContext& device() noexcept { return *device_; }

    [[nodiscard]] runtime::Executor<q35::ExecutionTypes>& executor() noexcept { return *executor_; }

    [[nodiscard]] q35::detail::RankProgramImpl& rank() noexcept {
        return observed_->implementation;
    }

    [[nodiscard]] ObservedRank& observer() noexcept { return *observed_; }

    std::shared_ptr<const q35::ModelDefinition> definition;
    q35::SessionPlan session_plan;

private:
    inline static std::atomic<std::uint64_t> next_epoch_{1};
    DeviceContext* device_  = nullptr;
    ObservedRank* observed_ = nullptr;
    std::unique_ptr<runtime::LocalExecutor<ObservedTypes>> executor_;
};

// Drives real completion events. Tests retain control over when to commit, cancel, inspect
// device state or submit competing work; only repetitive event pumping lives here.
template <class Session>
class EventDriver {
public:
    using DecisionHandler = std::function<runtime::DecisionInput(const runtime::DecisionRequest&)>;

    EventDriver(Session& session, NativeRankFixture& rank) : session_(session), rank_(rank) {}

    void synchronize() {
        rank_.device().synchronize();
        CUDA_CHECK(cudaStreamSynchronize(rank_.device().transfer_stream()));
    }

    template <class Result>
    Result wait(std::optional<runtime::ExecutionTicket> ticket = {},
                DecisionHandler decision                       = {}) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(2);
        for (;;) {
            for (auto it = events_.begin(); it != events_.end(); ++it) {
                if ((!ticket || it->ticket == *ticket) &&
                    std::holds_alternative<Result>(it->result)) {
                    auto result = std::move(std::get<Result>(it->result));
                    events_.erase(it);
                    return result;
                }
            }
            pump(decision);
            if (std::chrono::steady_clock::now() > deadline)
                throw std::runtime_error("native execution did not produce its expected event");
        }
    }

    void retire(runtime::ExecutionTicket ticket) {
        const auto found = members_.find(ticket.id);
        if (found == members_.end()) throw std::logic_error("unknown native result ticket");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(2);
        while (std::any_of(found->second.begin(), found->second.end(),
                           [&](auto member) { return session_.execution_pending(member); })) {
            pump({});
            if (std::chrono::steady_clock::now() > deadline)
                throw std::runtime_error("native execution did not retire");
        }
        members_.erase(found);
    }

    void idle() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(2);
        while (rank_.executor().has_pending_work() || session_.has_context_operation()) {
            pump({});
            if (std::chrono::steady_clock::now() > deadline)
                throw std::runtime_error("native worker did not become idle");
        }
    }

    q35::ContextProgress settle(bool cancelled = false) {
        if (cancelled) session_.cancel_context();
        return wait<q35::ContextProgress>();
    }

    std::optional<q35::ContextProgress> poll_context() {
        pump({});
        for (auto it = events_.begin(); it != events_.end(); ++it) {
            if (auto* result = std::get_if<q35::ContextProgress>(&it->result)) {
                auto value = std::move(*result);
                events_.erase(it);
                return value;
            }
        }
        return {};
    }

    q35::PrefillProgress prefill(q35::SequenceHandle sequence) {
        const auto ticket = session_.start_prefill(sequence);
        if (!ticket) throw std::logic_error("native prefill had no execution capacity");
        return wait<q35::PrefillProgress>(*ticket);
    }

    q35::ReplayProgress replay(q35::SequenceHandle sequence) {
        const auto ticket = session_.start_replay(sequence);
        if (!ticket) throw std::logic_error("native replay had no execution capacity");
        return wait<q35::ReplayProgress>(*ticket);
    }

    q35::PendingBatch decode(std::span<const q35::SequenceHandle> members,
                             std::span<const runtime::RoundBudget> budgets,
                             std::span<const bool> constrained = {},
                             DecisionHandler decision          = {}) {
        const auto ticket = session_.start_decode(members, budgets, constrained);
        if (!ticket) throw std::logic_error("native decode had no execution capacity");
        return wait<q35::PendingBatch>(*ticket, std::move(decision));
    }

    q35::CommitResult commit(q35::PendingBatch&& pending,
                             std::span<const runtime::CommitDecision> decisions) {
        const auto ticket = pending.ticket();
        auto result       = session_.commit(std::move(pending), decisions);
        retire(ticket);
        return result;
    }

    q35::FinishResult finish(q35::SequenceHandle sequence) {
        auto result = session_.finish(sequence);
        idle();
        return result;
    }

    q35::AbortResult abort(q35::SequenceHandle sequence) {
        auto result = session_.abort(sequence);
        idle();
        return result;
    }

    q35::ControlComplete control(std::span<const q35::SequenceHandle> members,
                                 std::span<const TokenId> tokens, std::uint32_t stride,
                                 std::span<const std::optional<std::uint32_t>> splits) {
        const auto ticket = session_.start_control(members, tokens, stride, splits);
        if (!ticket) throw std::logic_error("native control had no execution capacity");
        return wait<q35::ControlComplete>(*ticket);
    }

private:
    void pump(const DecisionHandler& decision) {
        if (auto event = session_.poll()) {
            if (const auto* request = std::get_if<runtime::DecisionRequest>(&event->result)) {
                if (!decision) throw std::logic_error("unexpected constrained decision request");
                session_.supply(event->ticket, decision(*request));
            } else if (!std::holds_alternative<runtime::DecisionConsumed>(event->result)) {
                if (!event->members.empty()) members_[event->ticket.id] = event->members;
                events_.push_back(std::move(*event));
            }
        } else {
            rank_.executor().flush();
            std::this_thread::yield();
        }
    }

    Session& session_;
    NativeRankFixture& rank_;
    std::deque<q35::SessionEvent> events_;
    std::map<std::uint64_t, std::vector<q35::SequenceHandle>> members_;
};

} // namespace ninfer::test::qwen3_5
