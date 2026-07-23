#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/mtp_round.h"
#include "ninfer/ops/scalar.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
void mtp_bridge_and_propose(State& state, const Tensor& next_token, const Tensor& previous_hidden,
                            std::int32_t position, std::span<const std::int32_t> rope_position,
                            bool build_proposal) {
    if (state.mtp_kv == nullptr || !state.io.mtp) {
        throw std::logic_error("MTP bridge requires MTP storage");
    }
    if (rope_position.size() != 3) {
        throw std::invalid_argument("MTP bridge requires one three-axis rope position");
    }
    state.work.reset();
    TextContext card(state.device, state.model, state.work, state.text_kv, state.gdn, state.io,
                     state.prefill_hidden, state.prefill_chunk, state.text_kv_base, state.mtp_kv);
    configure_text_card(card, state);

    Tensor position_view = state.io.speculative.target_positions.slice(0, 0, 1);
    ops::set_i32_scalar(position_view, position, state.device.stream);
    Tensor mtp_hidden         = state.io.mtp->ar_hidden;
    Tensor logits             = state.io.logits.slice(1, 0, 1);
    Tensor draft0             = state.io.speculative.draft_tokens.slice(0, 0, 1);
    Tensor rope_position_view = state.work.alloc(DType::I32, {1, 3});
    CUDA_CHECK(cudaMemcpyAsync(rope_position_view.data, rope_position.data(),
                               rope_position.size_bytes(), cudaMemcpyHostToDevice,
                               state.device.stream));
    const auto bridge_visible = static_cast<std::uint32_t>(position + 1);
    const ops::GqaExecutionEnvelope bridge_envelope{bridge_visible, bridge_visible};
    card.mtp_forward_batch(next_token, previous_hidden, position_view, bridge_envelope, mtp_hidden,
                           build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                           build_proposal ? &draft0 : nullptr, &rope_position_view);
    if (!build_proposal) { return; }

    ops::set_i32_scalar(state.io.mtp->position, position + 1, state.device.stream);
    for (int i = 1; i < state.io.speculative.draft_tokens.ne[0]; ++i) {
        Tensor previous_token = state.io.speculative.draft_tokens.slice(0, i - 1, 1);
        Tensor next_draft     = state.io.speculative.draft_tokens.slice(0, i, 1);
        Tensor next_hidden    = state.prefill_hidden.slice(1, i, 1);
        const auto visible    = static_cast<std::uint32_t>(position + i + 1);
        const ops::GqaExecutionEnvelope envelope{visible, visible};
        card.mtp_forward_ar_step(previous_token, state.io.mtp->ar_hidden, state.io.mtp->position,
                                 envelope, next_hidden, logits, next_draft);
        CUDA_CHECK(cudaMemcpyAsync(state.io.mtp->ar_hidden.data, next_hidden.data,
                                   state.io.mtp->ar_hidden.bytes(), cudaMemcpyDeviceToDevice,
                                   state.device.stream));
        ops::increment_i32_scalar(state.io.mtp->position, state.device.stream);
    }
}

auto mtp_body(State& state, std::uint32_t k, MtpGqaEnvelopes envelopes) {
    if (state.mtp_kv == nullptr || !state.io.mtp || k == 0 ||
        k != static_cast<std::uint32_t>(state.io.speculative.draft_tokens.ne[0])) {
        throw std::logic_error("MTP round storage does not match its configured window");
    }

    auto record = [&state, k, envelopes] {
        TextContext card(state.device, state.model, state.work, state.text_kv, state.gdn, state.io,
                         state.prefill_hidden, state.prefill_chunk, state.text_kv_base,
                         state.mtp_kv);
        configure_text_card(card, state);

        speculative_verify_and_accept(state, card, k, envelopes.target_verify);
        ops::assign_i32_scalar(state.io.pos, state.io.mtp->position, state.device.stream);

        const int columns = static_cast<int>(k) + 1;
        ops::mtp_prepare_alignment_ids(state.io.speculative.target_input_ids, state.io.token,
                                       state.io.speculative.accepted_drafts,
                                       state.io.mtp->alignment_ids, state.device.stream);
        Tensor mtp_hidden = state.prefill_hidden.slice(1, 0, columns);
        card.mtp_forward_batch(state.io.mtp->alignment_ids, state.io.verify_hidden,
                               state.io.speculative.target_positions, envelopes.batch, mtp_hidden,
                               -1, nullptr, nullptr);

        Tensor logits = state.io.logits.slice(1, 0, 1);
        Tensor draft0 = state.io.speculative.draft_tokens.slice(0, 0, 1);
        card.mtp_sample_from_hidden_row(mtp_hidden, state.io.speculative.accepted_drafts,
                                        state.io.mtp->ar_hidden, logits, draft0);
        for (std::uint32_t i = 1; i < k; ++i) {
            Tensor previous_token =
                state.io.speculative.draft_tokens.slice(0, static_cast<int>(i) - 1, 1);
            Tensor next_token  = state.io.speculative.draft_tokens.slice(0, static_cast<int>(i), 1);
            Tensor next_hidden = state.prefill_hidden.slice(1, static_cast<int>(i), 1);
            card.mtp_forward_ar_step(previous_token, state.io.mtp->ar_hidden,
                                     state.io.mtp->position, envelopes.ar[i - 1], next_hidden,
                                     logits, next_token);
            CUDA_CHECK(cudaMemcpyAsync(state.io.mtp->ar_hidden.data, next_hidden.data,
                                       state.io.mtp->ar_hidden.bytes(), cudaMemcpyDeviceToDevice,
                                       state.device.stream));
            ops::increment_i32_scalar(state.io.mtp->position, state.device.stream);
        }
    };
    return record;
}

void warm_capture_mtp_round(State& state, std::uint32_t k, MtpGqaEnvelopes envelopes,
                            const GraphPrepare& prepare, DecodeGraph& graph) {
    auto body = mtp_body(state, k, envelopes);
    warm_capture(state, graph, prepare, body);
}

void mtp_round(State& state, std::uint32_t k, MtpGqaEnvelopes envelopes, DecodeGraph* graph) {
    auto body = mtp_body(state, k, envelopes);
    run_prepared(state, graph, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
