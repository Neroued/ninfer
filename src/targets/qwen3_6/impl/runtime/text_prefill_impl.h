#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/linear.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scalar.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

void configure_text_card(TextContext& card, const State& state) {
    card.set_sampling(state.sampling);
    if (state.proposal_head == ProposalHead::Full) {
        card.set_proposal_head(nullptr, nullptr, 0);
        return;
    }
    if (card.proposal_head() == nullptr || card.proposal_head_ids() == nullptr ||
        card.proposal_head_n() <= 0) {
        throw std::runtime_error("optimized proposal head is unavailable");
    }
}

bool prefill_text(State& state, std::span<const TokenId> ids,
                  std::optional<std::uint32_t> snapshot_boundary, bool prepare_mtp) {
    TextContext card(state.device, state.model, state.work, state.text_kv, state.gdn, state.io,
                     state.prefill_hidden, state.prefill_chunk, state.text_kv_base,
                     prepare_mtp ? state.mtp_kv : nullptr);
    configure_text_card(card, state);
    card.set_boundary_hidden_output(state.boundary_hidden);
    card.set_prefill_snapshot_boundary(
        snapshot_boundary ? static_cast<std::int64_t>(*snapshot_boundary) : -1);
    const std::span<const int> prompt(ids.data(), ids.size());
    if (state.dflash != nullptr) {
        DFlashFeatureSink sink = dflash_feature_sink(
            state, [&state](const Tensor& features, const Tensor& positions, bool boundary) {
                ops::set_i32_scalar(state.dflash->commit_count, features.ne[1],
                                    state.device.stream);
                const auto count = static_cast<std::uint32_t>(features.ne[1]);
                dflash_append_context(state, features, positions, state.dflash->commit_count,
                                      {count, count});
                if (boundary) { state.dflash->save_boundary(state.device.stream); }
            });
        card.prefill(prompt, sink);
    } else if (state.diagnostic_text_tap != nullptr) {
        card.diagnostic_prefill(prompt, state.diagnostic_context, state.diagnostic_text_tap);
    } else {
        card.prefill(prompt);
    }
    return card.mtp_prompt_prepared();
}

MultimodalPrefillResult prefill_multimodal(State& state, const PreparedPromptData& prompt,
                                           const VisionPrefillPlan& plan,
                                           runtime::TransientRegion transient,
                                           std::optional<std::uint32_t> snapshot_boundary,
                                           bool prepare_mtp) {
    TextContext card(state.device, state.model, state.work, state.text_kv, state.gdn, state.io,
                     state.prefill_hidden, state.prefill_chunk, state.text_kv_base,
                     prepare_mtp ? state.mtp_kv : nullptr);
    configure_text_card(card, state);
    card.set_boundary_hidden_output(state.boundary_hidden);
    card.set_prefill_snapshot_boundary(
        snapshot_boundary ? static_cast<std::int64_t>(*snapshot_boundary) : -1);
    void* vision_tap_context =
        state.diagnostic_vision_tap != nullptr ? state.diagnostic_context : nullptr;
    VisionPrefillSession vision(state.device, state.model, state.work, prompt, plan, transient,
                                vision_tap_context, state.diagnostic_vision_tap);
    if (state.diagnostic_text_tap != nullptr) {
        card.diagnostic_prefill(prompt, state.text_kv_base, vision, state.diagnostic_context,
                                state.diagnostic_text_tap);
    } else {
        card.prefill(prompt, state.text_kv_base, vision);
    }
    if (card.last_prefill_chunk_length() == 0) {
        throw std::logic_error("multimodal prefill produced no final chunk");
    }
    return MultimodalPrefillResult{card.mtp_prompt_prepared(), card.last_prefill_chunk_length(),
                                   vision.elapsed_seconds()};
}

void sample_from_hidden(State& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose) {
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != TextConfig::hidden || hidden.ne[1] != 1 ||
        hidden.ne[2] != 1 || hidden.ne[3] != 1 || hidden.data == nullptr) {
        throw std::invalid_argument("sample_from_hidden requires BF16 [hidden,1]");
    }
    state.work.reset();
    Tensor logits = state.io.logits.slice(1, 0, 1);
    ops::linear(hidden, state.model.output_head, logits, state.work, state.device.stream);
    CUDA_CHECK(cudaMemcpyAsync(state.io.pos.data, &absolute_position, sizeof(absolute_position),
                               cudaMemcpyHostToDevice, state.device.stream));
    ops::sample(logits, state.io.token, TextConfig::token_domain, state.sampling,
                static_cast<const std::int32_t*>(state.io.pos.data), purpose, state.work,
                state.device.stream);
    state.work.reset();
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
