#pragma once

#include "runtime/engine/startup.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer::bench::qwen3_5 {

namespace model = models::qwen3_5;
using Instance  = runtime::InferenceInstance<model::RuntimeTypes>;
using Clock     = std::chrono::steady_clock;

struct RoundMeasurement {
    double wall_ms                = 0;
    double host_call_ms           = 0;
    std::uint32_t licensed_tokens = 0;
    std::array<SpeculativeStats, kMaximumConcurrency> stats{};
};

inline std::string device_name(int ordinal) {
    cudaDeviceProp properties{};
    const auto status = cudaGetDeviceProperties(&properties, ordinal);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    return properties.name;
}

class RoundFixture {
public:
    explicit RoundFixture(Instance& instance) : instance_(instance) {
        instance_.executor->bind_to_current_thread();
    }

    model::SequenceHandle bind(std::vector<TokenId> tokens, runtime::LaneId lane,
                               std::uint32_t output_tokens) {
        auto& session = *instance_.session;
        auto prompt   = instance_.frontend.prepare_tokens(std::move(tokens), false);
        runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens = output_tokens;
        options.allow_prefix_reuse      = false;
        auto plan                       = session.plan_request(std::move(prompt), options);
        const auto source               = session.inspect_source(plan, std::nullopt);
        if (!source || !session.start_binding(plan, lane, *source)) {
            throw std::runtime_error("benchmark root binding could not reserve its first unit");
        }
        auto binding = await<model::ContextProgress>();
        if (!binding.published || !binding.sequence ||
            binding.kind != model::ContextOperationKind::Bind) {
            throw std::runtime_error("benchmark root binding did not publish a sequence");
        }
        const auto sequence = *binding.sequence;
        for (;;) {
            const std::array<model::ExecutionUnit, 1> unit{
                model::ExecutionUnit{sequence, model::ExecutionUnitKind::Prefill, 0}};
            if (!session.reserve_units(unit)) {
                throw std::runtime_error("benchmark seed prefill unit could not be reserved");
            }
            const auto ticket = session.start_prefill(sequence);
            if (!ticket) throw std::runtime_error("benchmark seed execution slot is occupied");
            auto progress = await<model::PrefillProgress>(*ticket);
            if (!progress.complete) continue;
            if (!progress.pending || progress.pending->tokens().size() != 1) {
                throw std::runtime_error(
                    "benchmark seed prefill did not license exactly one token");
            }
            const std::array<runtime::CommitDecision, 1> decision{
                runtime::CommitDecision{.accepted_tokens = 1}};
            (void)session.commit(std::move(*progress.pending), decision);
            retire(*ticket, {&sequence, 1});
            return sequence;
        }
    }

    RoundMeasurement measure(std::span<const model::SequenceHandle> sequences,
                             std::uint32_t draft_tokens) {
        const auto batch = sequences.size();
        if (!batch || batch > kMaximumConcurrency)
            throw std::invalid_argument("invalid benchmark batch");
        auto& session = *instance_.session;
        std::array<runtime::RoundBudget, kMaximumConcurrency> budgets{};
        std::array<model::ExecutionUnit, kMaximumConcurrency> units{};
        for (std::size_t row = 0; row < batch; ++row) {
            budgets[row] = {draft_tokens + 1};
            units[row]   = {sequences[row], model::ExecutionUnitKind::Decode, draft_tokens + 1};
        }
        // Capacity policy is outside the complete execution interval.
        if (!session.reserve_units({units.data(), batch})) {
            throw std::runtime_error("benchmark decode batch could not be reserved");
        }
        host_call_ms_      = 0;
        const auto started = Clock::now();
        const auto ticket =
            call([&] { return session.start_decode(sequences, {budgets.data(), batch}); });
        if (!ticket) throw std::runtime_error("benchmark decode execution slot is occupied");
        auto pending = await<model::PendingBatch>(*ticket);
        if (pending.row_count() != batch ||
            (!pending.row_counts().empty() && pending.row_counts().size() != batch)) {
            throw std::runtime_error("benchmark round returned invalid compact membership");
        }
        std::array<runtime::CommitDecision, kMaximumConcurrency> decisions{};
        RoundMeasurement result;
        for (std::size_t row = 0; row < batch; ++row) {
            const auto count = pending.row_counts().empty() ? 1 : pending.row_counts()[row];
            if (count <= 0 || count > static_cast<std::int32_t>(draft_tokens + 1)) {
                throw std::runtime_error("benchmark round returned an invalid row extent");
            }
            decisions[row].accepted_tokens = static_cast<std::uint32_t>(count);
            result.licensed_tokens += static_cast<std::uint32_t>(count);
        }
        auto committed =
            call([&] { return session.commit(std::move(pending), {decisions.data(), batch}); });
        retire(*ticket, sequences);
        result.wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        result.host_call_ms = host_call_ms_;
        for (std::size_t row = 0; row < batch; ++row)
            result.stats[row] = std::move(committed.rows[row].speculative);
        return result;
    }

    model::AbortResult release(model::SequenceHandle sequence) {
        auto result = instance_.session->abort(sequence);
        if (result.status != runtime::ConsumeStatus::Consumed) {
            throw std::runtime_error("benchmark could not release its active sequence");
        }
        while (instance_.session->has_context_operation() ||
               instance_.executor->has_pending_work()) {
            if (instance_.session->poll())
                throw std::logic_error("unexpected result during release");
            instance_.executor->flush();
            std::this_thread::yield();
        }
        return result;
    }

private:
    template <class Function>
    std::invoke_result_t<Function> call(Function&& function) {
        const auto started = Clock::now();
        auto result        = function();
        host_call_ms_ += std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        return result;
    }

    template <class Result>
    Result await(std::optional<runtime::ExecutionTicket> ticket = {}) {
        for (;;) {
            instance_.executor->flush();
            auto event = call([&] { return instance_.session->poll(); });
            if (event) {
                if ((ticket && event->ticket != *ticket) ||
                    !std::holds_alternative<Result>(event->result)) {
                    throw std::logic_error("benchmark received an unrelated Session result");
                }
                return std::move(std::get<Result>(event->result));
            }
            std::this_thread::yield();
        }
    }

    void retire(runtime::ExecutionTicket ticket, std::span<const model::SequenceHandle> sequences) {
        const auto complete = await<runtime::CommitComplete>(ticket);
        if (complete.row_count != sequences.size() ||
            std::any_of(sequences.begin(), sequences.end(), [&](const auto sequence) {
                return instance_.session->execution_pending(sequence);
            })) {
            throw std::logic_error("benchmark commit did not retire its complete membership");
        }
    }

    Instance& instance_;
    double host_call_ms_ = 0;
};

} // namespace ninfer::bench::qwen3_5
