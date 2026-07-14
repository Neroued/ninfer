#include "runtime/generation/output_resolution.h"

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::runtime;

struct Candidate {
    StagedChoiceId id;
    std::uint32_t committed_tokens       = 0;
    RoundContinuation continuation       = RoundContinuation::Continue;
    FinishReason finish_reason           = FinishReason::None;
    OutputChannel channel                = OutputChannel::Content;
    std::uint64_t decoded_byte_cut_order = 0;
    std::uint32_t declaration_order      = 0;
};

struct Stage {
    std::vector<Candidate> values;

    std::span<const Candidate> candidates() const noexcept { return values; }

    StagedChoiceId choice_for(std::uint32_t tokens, RoundContinuation continuation,
                              FinishReason reason) const {
        for (const Candidate& candidate : values) {
            if (candidate.committed_tokens == tokens && candidate.continuation == continuation &&
                candidate.finish_reason == reason) {
                return candidate.id;
            }
        }
        throw std::logic_error("missing staged choice");
    }
};

Candidate choice(std::uint32_t id, std::uint32_t tokens, RoundContinuation continuation,
                 FinishReason reason, std::uint64_t byte_cut = 0, std::uint32_t declaration = 0) {
    return Candidate{.id                     = {id},
                     .committed_tokens       = tokens,
                     .continuation           = continuation,
                     .finish_reason          = reason,
                     .decoded_byte_cut_order = byte_cut,
                     .declaration_order      = declaration};
}

void require(bool condition) {
    if (!condition) { throw std::runtime_error("output resolution test failed"); }
}

} // namespace

int main() {
    const std::array<TokenId, 3> tokens{1, 2, 3};

    Stage stop_before_limit{{choice(1, 3, RoundContinuation::Continue, FinishReason::None),
                             choice(2, 3, RoundContinuation::Finish, FinishReason::OutputLimit),
                             choice(3, 2, RoundContinuation::Finish, FinishReason::StopToken)}};
    auto resolved = decide_resolution(stop_before_limit, tokens,
                                      GenerationBudget(3, FinishReason::OutputLimit));
    require(resolved.committed_tokens == 2 && resolved.finish_reason == FinishReason::StopToken &&
            resolved.staged_choice.value == 3);

    Stage string_beats_token{
        {choice(4, 3, RoundContinuation::Continue, FinishReason::None),
         choice(5, 2, RoundContinuation::Finish, FinishReason::StopToken),
         choice(6, 2, RoundContinuation::Finish, FinishReason::StopString, 9, 1)}};
    resolved = decide_resolution(string_beats_token, tokens,
                                 GenerationBudget(8, FinishReason::OutputLimit));
    require(resolved.finish_reason == FinishReason::StopString &&
            resolved.staged_choice.value == 6);

    Stage declaration_order{
        {choice(7, 3, RoundContinuation::Continue, FinishReason::None),
         choice(8, 2, RoundContinuation::Finish, FinishReason::StopString, 7, 3),
         choice(9, 2, RoundContinuation::Finish, FinishReason::StopString, 7, 1)}};
    resolved = decide_resolution(declaration_order, tokens,
                                 GenerationBudget(8, FinishReason::OutputLimit));
    require(resolved.staged_choice.value == 9);

    Stage limit{{choice(10, 3, RoundContinuation::Continue, FinishReason::None),
                 choice(11, 3, RoundContinuation::Finish, FinishReason::ContextCapacity)}};
    resolved = decide_resolution(limit, tokens, GenerationBudget(3, FinishReason::ContextCapacity));
    require(resolved.committed_tokens == 3 &&
            resolved.finish_reason == FinishReason::ContextCapacity &&
            resolved.staged_choice.value == 11);
}
