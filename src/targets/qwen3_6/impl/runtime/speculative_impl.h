#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/scalar.h"
#include "ninfer/ops/speculative_round.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

void target_verify(TextContext& card, State& state, const Tensor& ids, const Tensor& positions,
                   ops::GqaExecutionEnvelope envelope) {
    if (state.dflash != nullptr) {
        DFlashFeatureSink sink = dflash_feature_sink(state);
        card.target_verify(ids, positions, envelope, sink);
    } else if (state.diagnostic_text_tap != nullptr) {
        card.diagnostic_target_verify(ids, positions, envelope, state.diagnostic_context,
                                      state.diagnostic_text_tap);
    } else {
        card.target_verify(ids, positions, envelope);
    }
}

void speculative_verify_and_accept(State& state, TextContext& card, std::uint32_t draft_window,
                                   ops::GqaExecutionEnvelope target_envelope) {
    auto& spec = state.io.speculative;
    if (draft_window == 0 || draft_window != static_cast<std::uint32_t>(spec.draft_tokens.ne[0])) {
        throw std::logic_error(
            "speculative round storage does not match its configured draft window");
    }
    if (state.tail_hidden == nullptr) {
        throw std::logic_error("speculative round requires target tail-hidden storage");
    }

    ops::speculative_prepare_verify_inputs(state.io.token, spec.draft_tokens, state.io.pos,
                                           spec.target_input_ids, spec.target_positions,
                                           state.device.stream);
    target_verify(card, state, spec.target_input_ids, spec.target_positions, target_envelope);
    ops::speculative_accept_greedy_drafts(
        spec.target_argmax, state.io.logits, spec.draft_tokens, state.io.pos, state.io.token,
        spec.round_tokens, spec.produced_count, spec.accepted_drafts, spec.stats,
        TextConfig::token_domain, state.sampling, state.work, state.device.stream);
    ops::assign_i32_scalar(spec.accepted_drafts, state.io.gdn_initial_slot, state.device.stream);
    ops::speculative_select_accepted_hidden(state.io.verify_hidden, spec.accepted_drafts,
                                            *state.tail_hidden, state.device.stream);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
