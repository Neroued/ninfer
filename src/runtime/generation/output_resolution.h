#pragma once

#include "runtime/contract/types.h"
#include "runtime/generation/generation_budget.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace ninfer::runtime {

namespace detail {

template <class Candidate>
bool resolution_precedes(const Candidate& candidate, const Candidate& current) {
    if (candidate.committed_tokens != current.committed_tokens) {
        return candidate.committed_tokens < current.committed_tokens;
    }

    const bool candidate_string = candidate.finish_reason == FinishReason::StopString;
    const bool current_string   = current.finish_reason == FinishReason::StopString;
    if (candidate_string != current_string) { return candidate_string; }
    if (candidate_string) {
        if (candidate.decoded_byte_cut_order != current.decoded_byte_cut_order) {
            return candidate.decoded_byte_cut_order < current.decoded_byte_cut_order;
        }
        return candidate.declaration_order < current.declaration_order;
    }

    const bool candidate_token = candidate.finish_reason == FinishReason::StopToken;
    const bool current_token   = current.finish_reason == FinishReason::StopToken;
    return candidate_token && !current_token;
}

} // namespace detail

template <class StagedText>
OutputResolution decide_resolution(const StagedText& staged, std::span<const TokenId> tokens,
                                   const GenerationBudget& budget) {
    if (tokens.empty() || tokens.size() > budget.remaining()) {
        throw std::logic_error("staged round exceeds the generation budget");
    }

    using Candidate = std::remove_cv_t<typename decltype(staged.candidates())::element_type>;
    std::optional<Candidate> selected;
    for (const Candidate& candidate : staged.candidates()) {
        if (candidate.continuation != RoundContinuation::Finish ||
            (candidate.finish_reason != FinishReason::StopToken &&
             candidate.finish_reason != FinishReason::StopString)) {
            continue;
        }
        if (candidate.committed_tokens == 0 || candidate.committed_tokens > tokens.size()) {
            throw std::logic_error("decoder returned an invalid stop candidate");
        }
        if (!selected || detail::resolution_precedes(candidate, *selected)) {
            selected = candidate;
        }
    }

    if (tokens.size() == budget.remaining()) {
        Candidate limit;
        limit.id                     = staged.choice_for(static_cast<std::uint32_t>(tokens.size()),
                                                         RoundContinuation::Finish, budget.limit_reason());
        limit.committed_tokens       = static_cast<std::uint32_t>(tokens.size());
        limit.continuation           = RoundContinuation::Finish;
        limit.finish_reason          = budget.limit_reason();
        limit.decoded_byte_cut_order = std::numeric_limits<std::uint64_t>::max();
        limit.declaration_order      = std::numeric_limits<std::uint32_t>::max();
        if (!selected || detail::resolution_precedes(limit, *selected)) { selected = limit; }
    }

    if (selected) {
        return OutputResolution{.committed_tokens = selected->committed_tokens,
                                .continuation     = RoundContinuation::Finish,
                                .finish_reason    = selected->finish_reason,
                                .staged_choice    = selected->id};
    }

    const auto count = static_cast<std::uint32_t>(tokens.size());
    return OutputResolution{
        .committed_tokens = count,
        .continuation     = RoundContinuation::Continue,
        .finish_reason    = FinishReason::None,
        .staged_choice = staged.choice_for(count, RoundContinuation::Continue, FinishReason::None),
    };
}

template <class StagedText>
OutputResolution terminal_resolution(const StagedText& staged, FinishReason reason) {
    return OutputResolution{
        .committed_tokens = 0,
        .continuation     = RoundContinuation::Finish,
        .finish_reason    = reason,
        .staged_choice    = staged.choice_for(0, RoundContinuation::Finish, reason),
    };
}

} // namespace ninfer::runtime
