#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/kv_cache_append_prefix.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/swa.h"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

void require_dflash_state(const State& state) {
    if (state.dflash == nullptr || state.dflash_workspace == nullptr ||
        !state.model.dflash.has_value()) {
        throw std::logic_error("DFlash schedule requires DFlash weights, state, and workspace");
    }
}

template <class V>
DFlashFeatureSink dflash_feature_sink_impl(State& state,
                                           DFlashFeatureSink::PrefillConsumer consume_prefill) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash feature capture is unavailable for this target");
    } else {
        require_dflash_state(state);
        using Config = typename V::DFlashConfig;
        return DFlashFeatureSink{
            .features        = &state.dflash_workspace->target_features,
            .positions       = &state.dflash_workspace->feature_positions,
            .layers          = std::span<const int>(Config::target_feature_layers),
            .consume_prefill = std::move(consume_prefill),
        };
    }
}

template <class V>
void dflash_append_context_impl(State& state, const Tensor& features, const Tensor& positions,
                                const Tensor& commit_count,
                                ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash context append is unavailable for this target");
    } else {
        require_dflash_state(state);
        using Config     = typename V::DFlashConfig;
        const int tokens = features.ne[1];
        if (tokens <= 0 || features.dtype != DType::BF16 ||
            features.ne[0] != Config::feature_rows || positions.dtype != DType::I32 ||
            positions.ne[0] != tokens || commit_count.dtype != DType::I32 ||
            commit_count.ne[0] != 1) {
            throw std::invalid_argument("DFlash context append inputs are invalid");
        }

        state.work.reset();
        Tensor projected = state.work.alloc(DType::BF16, {Config::hidden, tokens});
        ops::linear(features, state.model.dflash->feature_projection, projected, state.work,
                    state.device.stream);
        Tensor context = state.work.alloc(DType::BF16, {Config::hidden, tokens});
        ops::rmsnorm(projected, state.model.dflash->context_norm, Config::rms_epsilon, false,
                     context, state.device.stream);

        for (int layer = 0; layer < Config::layers; ++layer) {
            auto layer_scope   = state.work.scope();
            const auto& weight = state.model.dflash->layers.at(static_cast<std::size_t>(layer));
            Tensor key_raw =
                state.work.alloc(DType::BF16, {Config::head_dim, Config::kv_heads, tokens});
            Tensor value =
                state.work.alloc(DType::BF16, {Config::head_dim, Config::kv_heads, tokens});
            Tensor key_flat   = key_raw.view({Config::kv_size, tokens});
            Tensor value_flat = value.view({Config::kv_size, tokens});
            ops::linear_pair(context, weight.context_key, weight.context_value, key_flat,
                             value_flat, state.work, state.device.stream);
            Tensor key =
                state.work.alloc(DType::BF16, {Config::head_dim, Config::kv_heads, tokens});
            ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                         state.device.stream);
            ops::rope(positions, Config::head_dim, Config::rope_theta, key, state.device.stream);
            if (layer < Config::local_layers) {
                ops::kv_cache_append_prefix(
                    key, value, positions, commit_count, envelope,
                    state.dflash->local_layer(static_cast<std::uint32_t>(layer)),
                    state.device.stream);
            } else {
                ops::kv_cache_append_prefix(key, value, positions, commit_count, envelope,
                                            state.dflash->full_layer(), state.device.stream);
            }
        }
        state.work.reset();
    }
}

template <class V>
void dflash_propose_impl(State& state, std::uint32_t k, DFlashEnvelopes envelopes) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash proposal is unavailable for this target");
    } else {
        require_dflash_state(state);
        using Config = typename V::DFlashConfig;
        if (k == 0 || k > static_cast<std::uint32_t>(state.io.speculative.draft_tokens.ne[0])) {
            throw std::invalid_argument("DFlash proposal window is invalid");
        }
        const int block = static_cast<int>(k + 1);
        state.work.reset();

        Tensor ids       = state.work.alloc(DType::I32, {block});
        Tensor positions = state.work.alloc(DType::I32, {block});
        ops::prepare_masked_block(state.io.token, state.io.pos, Config::mask_token, ids, positions,
                                  state.device.stream);
        Tensor residual = state.work.alloc(DType::BF16, {Config::hidden, block});
        ops::embedding(ids, state.model.token_embedding, residual, state.device.stream);

        for (int layer = 0; layer < Config::layers; ++layer) {
            const auto& weight = state.model.dflash->layers.at(static_cast<std::size_t>(layer));
            {
                auto attention_scope = state.work.scope();
                Tensor hidden        = state.work.alloc(DType::BF16, {Config::hidden, block});
                ops::rmsnorm(residual, weight.input_norm, Config::rms_epsilon, false, hidden,
                             state.device.stream);
                Tensor query_raw =
                    state.work.alloc(DType::BF16, {Config::head_dim, Config::query_heads, block});
                Tensor key_raw =
                    state.work.alloc(DType::BF16, {Config::head_dim, Config::kv_heads, block});
                Tensor value =
                    state.work.alloc(DType::BF16, {Config::head_dim, Config::kv_heads, block});
                Tensor query_flat = query_raw.view({Config::query_size, block});
                Tensor key_flat   = key_raw.view({Config::kv_size, block});
                Tensor value_flat = value.view({Config::kv_size, block});
                ops::attn_input_proj(hidden, weight.query_key_value, query_flat, key_flat,
                                     value_flat, state.work, state.device.stream);
                Tensor query =
                    state.work.alloc(DType::BF16, {Config::head_dim, Config::query_heads, block});
                Tensor key =
                    state.work.alloc(DType::BF16, {Config::head_dim, Config::kv_heads, block});
                ops::rmsnorm(query_raw, weight.query_norm, Config::rms_epsilon, false, query,
                             state.device.stream);
                ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                             state.device.stream);
                ops::rope(positions, Config::head_dim, Config::rope_theta, query, key,
                          state.device.stream);
                Tensor attention =
                    state.work.alloc(DType::BF16, {Config::head_dim, Config::query_heads, block});
                if (layer < Config::local_layers) {
                    ops::swa(query, key, value, positions, Config::attention_scale,
                             state.dflash->local_layer(static_cast<std::uint32_t>(layer)),
                             envelopes.local, state.work, attention, state.device.stream);
                } else {
                    ops::bidirectional_gqa_attention(query, key, value, state.io.pos,
                                                     Config::attention_scale,
                                                     state.dflash->full_layer(), envelopes.full,
                                                     state.work, attention, state.device.stream);
                }
                ops::linear_add(attention.view({Config::query_size, block}),
                                weight.attention_output, residual, state.work, state.device.stream);
            }
            {
                auto mlp_scope = state.work.scope();
                Tensor hidden  = state.work.alloc(DType::BF16, {Config::hidden, block});
                ops::rmsnorm(residual, weight.post_attention_norm, Config::rms_epsilon, false,
                             hidden, state.device.stream);
                Tensor intermediate = state.work.alloc(DType::BF16, {Config::intermediate, block});
                ops::linear_swiglu(hidden, weight.gate_up, intermediate, state.work,
                                   state.device.stream);
                ops::linear_add(intermediate, weight.down, residual, state.work,
                                state.device.stream);
            }
        }

        Tensor proposal_hidden_in = residual.slice(1, 1, static_cast<int>(k));
        Tensor proposal_hidden =
            state.work.alloc(DType::BF16, {Config::hidden, static_cast<int>(k)});
        ops::rmsnorm(proposal_hidden_in, state.model.dflash->final_norm, Config::rms_epsilon, false,
                     proposal_hidden, state.device.stream);
        Tensor drafts = state.io.speculative.draft_tokens.slice(0, 0, static_cast<int>(k));
        if (state.proposal_head == ProposalHead::Full) {
            Tensor logits = state.io.logits.slice(1, 0, static_cast<int>(k));
            ops::linear(proposal_hidden, state.model.output_head, logits, state.work,
                        state.device.stream);
            ops::argmax(logits, drafts, TextConfig::token_domain, state.device.stream);
        } else {
            if (!state.model.optimized_proposal.has_value()) {
                throw std::logic_error("optimized DFlash proposal head is unavailable");
            }
            const auto& proposal = *state.model.optimized_proposal;
            Tensor logits =
                state.work.alloc(DType::BF16, {V::draft_head_rows, static_cast<int>(k)});
            ops::linear(proposal_hidden, proposal.head, logits, state.work, state.device.stream);
            ops::argmax(logits, drafts, V::draft_head_rows, state.device.stream);
            ops::proposal_remap_token_ids(drafts,
                                          static_cast<const std::int32_t*>(proposal.token_ids.data),
                                          V::draft_head_rows, state.device.stream);
        }
        state.work.reset();
    }
}

} // namespace

DFlashFeatureSink dflash_feature_sink(State& state,
                                      DFlashFeatureSink::PrefillConsumer consume_prefill) {
    return dflash_feature_sink_impl<Variant>(state, std::move(consume_prefill));
}

void dflash_append_context(State& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_count,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    dflash_append_context_impl<Variant>(state, features, positions, commit_count, envelope);
}

void dflash_propose(State& state, std::uint32_t k, DFlashEnvelopes envelopes) {
    dflash_propose_impl<Variant>(state, k, envelopes);
}

namespace {

auto dflash_initial_body(State& state, std::uint32_t k, ops::GqaExecutionEnvelope target_envelope) {
    return [&state, k, target_envelope] {
        TextContext card(state.device, state.model, state.work, state.text_kv, state.gdn, state.io,
                         state.prefill_hidden, state.prefill_chunk, state.text_kv_base);
        configure_text_card(card, state);
        speculative_verify_and_accept(state, card, k, target_envelope);
    };
}

auto dflash_steady_body(State& state, std::uint32_t k, DFlashEnvelopes envelopes,
                        ops::GqaExecutionEnvelope target_envelope) {
    return [&state, k, envelopes, target_envelope] {
        const int block  = static_cast<int>(k + 1);
        Tensor features  = state.dflash_workspace->target_features.slice(1, 0, block);
        Tensor positions = state.dflash_workspace->feature_positions.slice(0, 0, block);
        dflash_append_context(state, features, positions, state.io.speculative.produced_count,
                              envelopes.append);
        dflash_propose(state, k, envelopes);
        TextContext card(state.device, state.model, state.work, state.text_kv, state.gdn, state.io,
                         state.prefill_hidden, state.prefill_chunk, state.text_kv_base);
        configure_text_card(card, state);
        speculative_verify_and_accept(state, card, k, target_envelope);
    };
}

} // namespace

void warm_capture_dflash_initial_round(State& state, std::uint32_t k,
                                       ops::GqaExecutionEnvelope target_envelope,
                                       const GraphPrepare& prepare, DecodeGraph& graph) {
    auto body = dflash_initial_body(state, k, target_envelope);
    warm_capture(state, graph, prepare, body);
}

void dflash_initial_round(State& state, std::uint32_t k, ops::GqaExecutionEnvelope target_envelope,
                          DecodeGraph* graph) {
    auto body = dflash_initial_body(state, k, target_envelope);
    run_prepared(state, graph, body);
}

void warm_capture_dflash_steady_round(State& state, std::uint32_t k, DFlashEnvelopes envelopes,
                                      ops::GqaExecutionEnvelope target_envelope,
                                      const GraphPrepare& prepare, DecodeGraph& graph) {
    auto body = dflash_steady_body(state, k, envelopes, target_envelope);
    warm_capture(state, graph, prepare, body);
}

void dflash_steady_round(State& state, std::uint32_t k, DFlashEnvelopes envelopes,
                         ops::GqaExecutionEnvelope target_envelope, DecodeGraph* graph) {
    auto body = dflash_steady_body(state, k, envelopes, target_envelope);
    run_prepared(state, graph, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
