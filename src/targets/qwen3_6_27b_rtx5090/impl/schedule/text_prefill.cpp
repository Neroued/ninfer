#include "targets/qwen3_6_27b_rtx5090/impl/schedule/schedule.h"

#include "ninfer/ops/linear.h"
#include "ninfer/ops/sampling.h"
#include "targets/qwen3_6_27b_rtx5090/impl/config.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule {

void configure_text_card(TextContext& card, const State& state) {
    card.set_sampling(state.sampling);
    if (state.proposal_head == ProposalHead::Full) {
        card.set_lm_head_draft(nullptr, nullptr, 0);
        return;
    }
    if (card.lm_head_draft() == nullptr || card.lm_head_draft_ids() == nullptr ||
        card.lm_head_draft_n() <= 0) {
        throw std::runtime_error("optimized MTP proposal head is unavailable");
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
    if (state.diagnostic_text_tap != nullptr) {
        card.diagnostic_prefill(prompt, state.diagnostic_context, state.diagnostic_text_tap);
    } else {
        card.prefill(prompt);
    }
    return card.mtp_prompt_prepared();
}

bool prefill_multimodal(State& state, const PreparedPromptData& prompt,
                        const Tensor& visual_embeddings, bool prepare_mtp) {
    TextContext card(state.device, state.model, state.work, state.text_kv, state.gdn, state.io,
                     state.prefill_hidden, state.prefill_chunk, state.text_kv_base,
                     prepare_mtp ? state.mtp_kv : nullptr);
    configure_text_card(card, state);
    card.set_boundary_hidden_output(nullptr);
    card.set_prefill_snapshot_boundary(-1);
    if (state.diagnostic_text_tap != nullptr) {
        card.diagnostic_prefill(prompt, visual_embeddings, state.diagnostic_context,
                                state.diagnostic_text_tap);
    } else {
        card.prefill(prompt, visual_embeddings);
    }
    return card.mtp_prompt_prepared();
}

void sample_from_hidden(State& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose) {
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != TextConfig::hidden || hidden.ne[1] != 1 ||
        hidden.ne[2] != 1 || hidden.ne[3] != 1 || hidden.data == nullptr) {
        throw std::invalid_argument("sample_from_hidden requires BF16 [5120,1]");
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

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule
