#include "targets/qwen3_6_27b_rtx5090/impl/schedule/schedule.h"

#include "targets/qwen3_6_27b_rtx5090/impl/schedule/ops.h"
#include "targets/qwen3_6_27b_rtx5090/impl/config.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule {
namespace {

void copy_i32(const Tensor& source, Tensor& destination, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void set_i32(Tensor& destination, std::int32_t value, cudaStream_t stream) {
    CUDA_CHECK(
        cudaMemcpyAsync(destination.data, &value, sizeof(value), cudaMemcpyHostToDevice, stream));
}

void target_verify(TextContext& card, State& state, const Tensor& ids, const Tensor& positions) {
    if (state.diagnostic_text_tap != nullptr) {
        card.diagnostic_target_verify(ids, positions, state.diagnostic_context,
                                      state.diagnostic_text_tap);
    } else {
        card.target_verify(ids, positions);
    }
}

} // namespace

void mtp_bridge_and_propose(State& state, const Tensor& next_token, const Tensor& previous_hidden,
                            std::int32_t position, bool build_proposal) {
    if (state.mtp_kv == nullptr) { throw std::logic_error("MTP bridge requires MTP storage"); }
    TextContext card(state.device, state.model, state.work, state.text_kv, state.gdn, state.io,
                     state.prefill_chunk, state.mtp_kv);
    configure_text_card(card, state);

    Tensor position_view = state.io.positions.slice(0, 0, 1);
    set_i32(position_view, position, state.device.stream);
    Tensor mtp_hidden = state.io.mtp_ar_hidden;
    Tensor logits     = state.io.logits.slice(1, 0, 1);
    Tensor draft0     = state.io.drafts.slice(0, 0, 1);
    card.mtp_forward_batch(next_token, previous_hidden, position_view, mtp_hidden,
                           build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                           build_proposal ? &draft0 : nullptr);
    if (!build_proposal) { return; }

    set_i32(state.io.ar_pos, position + 1, state.device.stream);
    for (int i = 1; i < state.io.drafts.ne[0]; ++i) {
        Tensor previous_token = state.io.drafts.slice(0, i - 1, 1);
        Tensor next_draft     = state.io.drafts.slice(0, i, 1);
        Tensor next_hidden    = state.io.prefill_hidden.slice(1, i, 1);
        card.mtp_forward_ar_step(previous_token, state.io.mtp_ar_hidden, state.io.ar_pos,
                                 next_hidden, logits, next_draft);
        CUDA_CHECK(cudaMemcpyAsync(state.io.mtp_ar_hidden.data, next_hidden.data,
                                   state.io.mtp_ar_hidden.bytes(), cudaMemcpyDeviceToDevice,
                                   state.device.stream));
        kernels::mtp_increment_i32(state.io.ar_pos, state.device.stream);
    }
}

template <class Body>
void run_prepared(State& state, DecodeGraph* graph, Body&& body) {
    if (graph != nullptr) {
        if (!graph->ready()) { throw std::logic_error("decode graph was not prepared at load time"); }
        graph->launch(state.device.stream);
    } else {
        body();
    }
}

template <class Body>
void warm_capture(State& state, DecodeGraph& graph, Body&& body) {
    body();
    state.device.synchronize();
    graph.capture(state.device.stream, body);
    graph.launch(state.device.stream);
    state.device.synchronize();
}

auto ordinary_body(State& state, bool align_mtp) {
    auto record = [&state, align_mtp] {
        TextContext card(state.device, state.model, state.work, state.text_kv, state.gdn, state.io,
                         state.prefill_chunk, align_mtp ? state.mtp_kv : nullptr);
        configure_text_card(card, state);

        Tensor verify_id = state.io.verify_ids.slice(0, 0, 1);
        Tensor position  = state.io.positions.slice(0, 0, 1);
        copy_i32(state.io.token, verify_id, state.device.stream);
        copy_i32(state.io.pos, position, state.device.stream);
        target_verify(card, state, verify_id, position);

        Tensor logits = state.io.logits.slice(1, 0, 1);
        kernels::sample(logits, state.io.token, TextConfig::token_domain, state.sampling,
                        static_cast<const std::int32_t*>(state.io.pos.data),
                        kernels::kSamplePurposeDecode, state.device.stream);

        if (align_mtp) {
            Tensor hidden     = state.io.verify_hidden.slice(1, 0, 1);
            Tensor mtp_hidden = state.io.mtp_ar_hidden;
            card.mtp_forward_batch(state.io.token, hidden, position, mtp_hidden, -1, nullptr,
                                   nullptr);
        }

        kernels::mtp_increment_i32(state.io.pos, state.device.stream);
        kernels::mtp_increment_i32(state.io.rope_pos, state.device.stream);
        kernels::mtp_reset_gdn_initial_slot(state.io.gdn_initial_slot, state.device.stream);
        if (state.mtp_kv != nullptr) {
            kernels::mtp_count_fallback_step(state.io.stats, state.device.stream);
        }
    };
    return record;
}

auto mtp_body(State& state, std::uint32_t k) {
    if (state.mtp_kv == nullptr || k == 0 ||
        k != static_cast<std::uint32_t>(state.io.drafts.ne[0])) {
        throw std::logic_error("MTP round storage does not match its configured window");
    }

    auto record = [&state, k] {
        TextContext card(state.device, state.model, state.work, state.text_kv, state.gdn, state.io,
                         state.prefill_chunk, state.mtp_kv);
        configure_text_card(card, state);

        kernels::mtp_prepare_verify_inputs(state.io.token, state.io.drafts, state.io.pos,
                                           state.io.window_base, state.io.verify_ids,
                                           state.io.positions, state.device.stream);
        target_verify(card, state, state.io.verify_ids, state.io.positions);
        kernels::mtp_accept_tokens(
            state.io.target_tokens, state.io.logits, state.io.drafts, state.io.pos, state.io.token,
            state.io.sampled_out, state.io.num_sampled, state.io.accepted, state.io.ar_pos,
            state.io.stats, TextConfig::token_domain, state.sampling, state.device.stream);
        kernels::mtp_set_gdn_initial_slot_from_accepted(
            state.io.accepted, state.io.gdn_initial_slot, state.device.stream);

        const int columns = static_cast<int>(k) + 1;
        kernels::mtp_prepare_shifted_ids(state.io.verify_ids, state.io.token, state.io.accepted,
                                         state.io.shifted_ids, state.device.stream);
        Tensor mtp_hidden = state.io.prefill_hidden.slice(1, 0, columns);
        card.mtp_forward_batch(state.io.shifted_ids, state.io.verify_hidden, state.io.positions,
                               mtp_hidden, -1, nullptr, nullptr);

        Tensor logits = state.io.logits.slice(1, 0, 1);
        Tensor draft0 = state.io.drafts.slice(0, 0, 1);
        card.mtp_sample_from_hidden_row(mtp_hidden, state.io.accepted, state.io.mtp_ar_hidden,
                                        logits, draft0);
        for (std::uint32_t i = 1; i < k; ++i) {
            Tensor previous_token = state.io.drafts.slice(0, static_cast<int>(i) - 1, 1);
            Tensor next_token     = state.io.drafts.slice(0, static_cast<int>(i), 1);
            Tensor next_hidden    = state.io.prefill_hidden.slice(1, static_cast<int>(i), 1);
            card.mtp_forward_ar_step(previous_token, state.io.mtp_ar_hidden, state.io.ar_pos,
                                     next_hidden, logits, next_token);
            CUDA_CHECK(cudaMemcpyAsync(state.io.mtp_ar_hidden.data, next_hidden.data,
                                       state.io.mtp_ar_hidden.bytes(), cudaMemcpyDeviceToDevice,
                                       state.device.stream));
            kernels::mtp_increment_i32(state.io.ar_pos, state.device.stream);
        }
    };
    return record;
}

void warm_capture_ordinary_round(State& state, bool align_mtp, DecodeGraph& graph) {
    auto body = ordinary_body(state, align_mtp);
    warm_capture(state, graph, body);
}

void ordinary_round(State& state, bool align_mtp, DecodeGraph* graph) {
    auto body = ordinary_body(state, align_mtp);
    run_prepared(state, graph, body);
}

void warm_capture_mtp_round(State& state, std::uint32_t k, DecodeGraph& graph) {
    auto body = mtp_body(state, k);
    warm_capture(state, graph, body);
}

void mtp_round(State& state, std::uint32_t k, DecodeGraph* graph) {
    auto body = mtp_body(state, k);
    run_prepared(state, graph, body);
}

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule
