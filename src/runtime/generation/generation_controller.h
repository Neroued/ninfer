#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "runtime/generation/generation_budget.h"
#include "runtime/generation/generation_guard.h"
#include "runtime/generation/output_publisher.h"

#include <chrono>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::runtime {

struct ControllerResult {
    GenerationSummary summary;
    std::vector<TokenId> generated_token_ids;
    std::string content;
    std::string reasoning;
    double prefill_seconds = 0.0;
    double decode_seconds  = 0.0;
    double total_seconds   = 0.0;
};

template <class Program, class PreparedPrompt, class OutputSession, class RequestMemory>
ControllerResult run_one(Program& program, PreparedPrompt prompt, OutputSession output,
                         RequestMemory& request_memory, const RequestOptions& options,
                         const CancellationView& cancellation, OutputSink* sink) {
    using Clock            = std::chrono::steady_clock;
    const auto total_start = Clock::now();

    ControllerResult result;
    OutputPublisher publisher(sink);

    const auto finish_without_execution = [&](FinishReason reason) {
        result.summary.finish_reason = reason;
        result.content               = publisher.take_content();
        result.reasoning             = publisher.take_reasoning();
        result.total_seconds = std::chrono::duration<double>(Clock::now() - total_start).count();
        return std::move(result);
    };

    if (cancellation.requested()) { return finish_without_execution(FinishReason::Cancelled); }
    if (options.execution.requested_output_tokens == 0) {
        return finish_without_execution(FinishReason::OutputLimit);
    }

    auto plan                             = program.plan_request(prompt, options.execution);
    const RequestPlanSummary plan_summary = plan.summary();
    if (plan_summary.effective_output_tokens == 0) {
        return finish_without_execution(plan_summary.effective_limit_reason);
    }
    request_memory.ensure(plan_summary.transient_bytes, plan_summary.transient_alignment);

    GenerationBudget budget(plan_summary.effective_output_tokens,
                            plan_summary.effective_limit_reason);
    result.generated_token_ids.reserve(plan_summary.effective_output_tokens);
    GenerationGuard guard(program);

    const auto finish_result = [&](FinishReason reason) {
        result.summary.finish_reason = reason;
        result.content               = publisher.take_content();
        result.reasoning             = publisher.take_reasoning();
        result.total_seconds = std::chrono::duration<double>(Clock::now() - total_start).count();
        return std::move(result);
    };

    const auto resolve_and_publish = [&](GeneratedRound round) -> std::optional<ControllerResult> {
        if (cancellation.requested()) {
            (void)output.preview_terminal(FinishReason::Cancelled);
            program.abort_request();
            auto deltas = output.commit_preview();
            publisher.publish(std::move(deltas));
            return finish_result(FinishReason::Cancelled);
        }

        const std::span<const TokenId> tokens = round.tokens;
        const OutputDecision decision =
            output.preview(tokens, budget.remaining(), budget.limit_reason());

        result.generated_token_ids.insert(
            result.generated_token_ids.end(), tokens.begin(),
            tokens.begin() + static_cast<std::ptrdiff_t>(decision.accepted_tokens));

        if (!decision.finished()) {
            if (decision.accepted_tokens != tokens.size()) {
                throw std::logic_error("a continuing round must commit every licensed token");
            }
        }
        program.resolve_pending(decision.accepted_tokens, decision.finished());

        auto deltas = output.commit_preview();
        budget.commit(decision.accepted_tokens);
        publisher.publish(std::move(deltas));

        if (decision.finished()) {
            return finish_result(decision.finish_reason);
        }
        return std::nullopt;
    };

    if (cancellation.requested()) { return finish_without_execution(FinishReason::Cancelled); }

    const auto prefill_start = Clock::now();
    auto first = program.begin(std::move(prompt), std::move(plan), request_memory.region());
    result.prefill_seconds = std::chrono::duration<double>(Clock::now() - prefill_start).count();
    result.summary.begin   = first.summary;
    guard.arm();
    if (auto done = resolve_and_publish(first.round)) {
        guard.complete();
        return std::move(*done);
    }

    while (budget.remaining() != 0) {
        if (cancellation.requested()) {
            (void)output.preview_terminal(FinishReason::Cancelled);
            program.finish_active();
            auto deltas = output.commit_preview();
            publisher.publish(std::move(deltas));
            guard.complete();
            return finish_result(FinishReason::Cancelled);
        }

        const auto decode_start = Clock::now();
        auto round              = program.decode_round(budget.round_budget());
        result.decode_seconds += std::chrono::duration<double>(Clock::now() - decode_start).count();
        if (auto done = resolve_and_publish(round)) {
            guard.complete();
            return std::move(*done);
        }
    }

    throw std::logic_error("generation budget reached an unresolved zero state");
}

} // namespace ninfer::runtime
