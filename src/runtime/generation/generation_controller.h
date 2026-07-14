#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "runtime/generation/generation_budget.h"
#include "runtime/generation/generation_guard.h"
#include "runtime/generation/output_publisher.h"
#include "runtime/generation/output_resolution.h"

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
    result.summary.committed_output_tokens = 0;
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

    const auto finish_result = [&](FinishReason reason,
                                   std::optional<FinishDisposition> disposition) {
        result.summary.committed_output_tokens =
            static_cast<std::uint32_t>(result.generated_token_ids.size());
        result.summary.finish_reason = reason;
        result.summary.sequence      = disposition;
        result.content               = publisher.take_content();
        result.reasoning             = publisher.take_reasoning();
        result.total_seconds = std::chrono::duration<double>(Clock::now() - total_start).count();
        return std::move(result);
    };

    const auto resolve_and_publish = [&](auto pending) -> std::optional<ControllerResult> {
        if (cancellation.requested()) {
            auto staged = output.stage_terminal(FinishReason::Cancelled);
            const OutputResolution resolution =
                terminal_resolution(staged, FinishReason::Cancelled);
            auto decoder_commit = output.prepare_commit(std::move(staged), resolution);
            std::move(pending).discard();
            auto deltas = output.commit(std::move(decoder_commit));
            publisher.publish(std::move(deltas));
            return finish_result(FinishReason::Cancelled, FinishDisposition::Invalid);
        }

        const std::span<const TokenId> tokens = pending.tokens();
        auto staged                           = output.stage(tokens);
        const OutputResolution resolution     = decide_resolution(staged, tokens, budget);
        auto decoder_commit = output.prepare_commit(std::move(staged), resolution);

        result.generated_token_ids.insert(
            result.generated_token_ids.end(), tokens.begin(),
            tokens.begin() + static_cast<std::ptrdiff_t>(resolution.committed_tokens));

        std::optional<FinishDisposition> disposition;
        if (resolution.continuation == RoundContinuation::Continue) {
            if (resolution.committed_tokens != tokens.size()) {
                throw std::logic_error("a continuing round must commit every licensed token");
            }
            std::move(pending).commit_all();
        } else {
            disposition = std::move(pending).commit_prefix_and_finish(resolution.committed_tokens);
        }

        auto deltas = output.commit(std::move(decoder_commit));
        budget.commit(resolution.committed_tokens);
        publisher.publish(std::move(deltas));

        if (resolution.continuation == RoundContinuation::Finish) {
            return finish_result(resolution.finish_reason, disposition);
        }
        return std::nullopt;
    };

    if (cancellation.requested()) { return finish_without_execution(FinishReason::Cancelled); }

    const auto prefill_start = Clock::now();
    auto first = program.begin(std::move(prompt), std::move(plan), request_memory.region());
    result.prefill_seconds = std::chrono::duration<double>(Clock::now() - prefill_start).count();
    result.summary.begin   = first.summary;
    guard.arm();
    if (auto done = resolve_and_publish(std::move(first.round))) {
        guard.complete();
        return std::move(*done);
    }

    while (budget.remaining() != 0) {
        if (cancellation.requested()) {
            auto staged = output.stage_terminal(FinishReason::Cancelled);
            const OutputResolution resolution =
                terminal_resolution(staged, FinishReason::Cancelled);
            auto decoder_commit = output.prepare_commit(std::move(staged), resolution);
            program.finish_active();
            auto deltas = output.commit(std::move(decoder_commit));
            publisher.publish(std::move(deltas));
            guard.complete();
            return finish_result(FinishReason::Cancelled, FinishDisposition::Resident);
        }

        const auto decode_start = Clock::now();
        auto round              = program.decode_round(budget.round_budget());
        result.decode_seconds += std::chrono::duration<double>(Clock::now() - decode_start).count();
        if (auto done = resolve_and_publish(std::move(round))) {
            guard.complete();
            return std::move(*done);
        }
    }

    throw std::logic_error("generation budget reached an unresolved zero state");
}

} // namespace ninfer::runtime
