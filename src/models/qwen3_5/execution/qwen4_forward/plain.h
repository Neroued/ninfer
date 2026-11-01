#pragma once

// The plain residual (no hyper-connections): the Qwen3.5 block, docs/maintainer/qwen3_5-model.md.
// One [H,T] residual x per rank (replicated at tp 2). Each block: the mixer normalizes x by
// input_norm and adds its output projection into x; the FFN normalizes x by post_attention_norm
// and adds its down projection into x. At tp 2 each closing projection is row-parallel with the
// residual folded into its all-reduce (linear_add_row_parallel), so both ranks keep the
// identical BF16 residual. The carried hidden is the final-normed residual, which the output
// head reads directly (head() is then a view). A dense Qwen3.5 model takes the 27B's fused routes
// in every block (Parameters).
// At tp 2 a prompt chunk overlaps its exchanges with compute, with the same bits as one call:
//   * the half-chunk overlap (prefill_overlap): the chunk's two halves are two consecutive
//     prompt chunks (PlainHalf: column views, positions, causal envelope, GDN slots - half B
//     continues from the state half A published) issued block by block in the order mixer A,
//     mixer B, FFN A, FFN B, next layer's mixer A, ... Each block begins its closing all-reduce
//     on its half's lane of the split exchange (lane_) and the half's next block finishes it
//     first, so one half's exchange moves while the other half's block computes. A feature tap
//     gets a layer's output per half once its closing exchange has finished, before the half's
//     next block rewrites the columns;
//   * otherwise the slices (tp2/prefill_slices.h): the mixer core runs on the whole chunk, its
//     output projection begins slice s's all-reduce on lane s, the FFN of slice s waits for lane
//     s only and begins its own on lane s again;
//   * otherwise one all-reduce per block.
// Captured tp 2 decode stages the L2 prefetch hints of the block's exchanges
// (tp2/exchange_prefetch.h).
//
// Member definitions of Qwen4Forward (qwen4_forward.h), included only by qwen4_forward.cpp, which
// instantiates them for execution and for planning.

#include "models/qwen3_5/execution/qwen4_forward.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/scatter.h"

#include <exception>
#include <stdexcept>
#include <string>
#include <variant>

namespace ninfer::models::qwen3_5::execution {

template <class Arena>
void Qwen4Forward<Arena>::run_plain(const RankTensors& ids, RankTensors& out,
                                    const VisualColumns* visual) {
    const std::int32_t T     = columns();
    const std::int32_t split = prefill_overlap(T);
    slices_                  = split == 0 ? prefill_slices(T) : PrefillSlicePlan{};
    auto scope               = scopes();
    RankTensors normed       = wide_buffers(out, T);
    RankTensors x            = alloc(DType::BF16, {hidden(), T});
    RankTensors staging;
    if (width_ == 2) { staging = alloc(DType::BF16, {hidden(), T}); }
    each([&](int r) {
        if constexpr (kExecute) {
            ops::embedding(ids[at(r)], parameters(r).text.token_embedding, x[at(r)], s(r));
            if (visual != nullptr) {
                ops::scatter(visual->embeddings[at(r)], visual->indices[at(r)], x[at(r)], s(r));
            }
        }
    });
    if constexpr (kExecute) {
        if (feature_tap_ != nullptr) { feature_tap_->begin(x); }
    }
    std::array<PlainHalf, 2> halves;
    if (split > 0) { halves = plain_halves(x, staging, split); }
    for (std::size_t layer = 0; layer < config_.layer_types.size(); ++layer) {
        try {
            if (split > 0) {
                plain_half_blocks(layer, halves);
            } else {
                plain_block(layer, x, staging);
            }
        } catch (const std::exception& error) {
            abandon_lanes(halves);
            throw std::runtime_error("text/layers/" + std::to_string(layer) + " " + phase() +
                                     (split > 0 ? " overlap" : "") + (width_ == 2 ? " tp2" : "") +
                                     " columns=" + std::to_string(T) + ": " + error.what());
        }
    }
    if (split > 0) {
        // The last layer's exchanges (and its taps) complete before the final norm reads x.
        try {
            finish_half(halves[0], 0);
            finish_half(halves[1], 1);
        } catch (...) {
            abandon_lanes(halves);
            throw;
        }
    }
    each([&](int r) {
        if constexpr (kExecute) {
            ops::rmsnorm(x[at(r)], qwen4(r).final_norm, config_.rms_norm_eps, true, normed[at(r)],
                         s(r));
        }
    });
    if constexpr (kExecute) {
        if (Qwen4Taps* taps = prompt_taps(-1)) { capture_taps(*taps, -1, normed); }
    }
}

template <class Arena>
void Qwen4Forward<Arena>::plain_mixer(std::size_t layer, RankTensors& x,
                                      const RankTensors& staging) {
    const auto compact = dimension(config_.compact_layer_indices[layer]);
    if (config_.layer_types[layer] == MixerKind::FullAttention) {
        dense_attention(layer, x, x, compact, staging);
    } else {
        gdn(layer, x, x, compact, staging);
    }
}

template <class Arena>
void Qwen4Forward<Arena>::plain_block(std::size_t layer, RankTensors& x,
                                      const RankTensors& staging) {
    // Captured tp 2 decode: each block's closing all-reduce prefetches the weights the GEMM
    // after it streams (tp2/exchange_prefetch.h; timings only).
    const std::size_t prefetch = exchange_prefetch(x);
    {
        auto mixer_scope = scopes();
        if (prefetch != 0) { stage_prefetch(x, ffn_input_hint(layer, prefetch)); }
        plain_mixer(layer, x, staging);
    }
    {
        auto ffn_scope = scopes();
        if (prefetch != 0) { stage_prefetch(x, next_input_hint(layer, prefetch)); }
        plain_ffn(layer, x, staging);
    }
    finish_slices();
    if constexpr (kExecute) {
        if (feature_tap_ != nullptr) { feature_tap_->capture_layer(static_cast<int>(layer), x); }
        if (Qwen4Taps* taps = prompt_taps(static_cast<std::int32_t>(layer))) {
            capture_taps(*taps, static_cast<std::int32_t>(layer), x);
        }
    }
}

template <class Arena>
void Qwen4Forward<Arena>::plain_half_blocks(std::size_t layer, std::array<PlainHalf, 2>& halves) {
    for (int h = 0; h < 2; ++h) {
        half_block(halves[at(h)], h,
                   [&](PlainHalf& half) { plain_mixer(layer, half.x, half.staging); });
    }
    for (int h = 0; h < 2; ++h) {
        half_block(
            halves[at(h)], h, [&](PlainHalf& half) { plain_ffn(layer, half.x, half.staging); },
            static_cast<int>(layer));
    }
}

template <class Arena>
std::size_t Qwen4Forward<Arena>::exchange_prefetch(const RankTensors& x) const {
    if constexpr (kExecute) {
        if (width_ == 2 && !c(0).prefill) {
            return exchange_prefetch_budget(*tp_.events, *tp_.execution, x[0].bytes());
        }
    } else {
        (void)x;
    }
    return 0;
}

template <class Arena>
void Qwen4Forward<Arena>::stage_prefetch(const RankTensors& x,
                                         const std::array<ExchangePrefetch, 2>& hint) {
    if constexpr (kExecute) {
        stage_exchange_prefetch(*tp_.events, *tp_.execution, x[0].bytes(), hint);
    } else {
        (void)x, (void)hint;
    }
}

template <class Arena>
std::int32_t Qwen4Forward<Arena>::prefill_overlap(std::int32_t T) const {
    if constexpr (kExecute) {
        const Qwen4Columns& col     = c(0);
        const Qwen4Runtime* runtime = state(0).runtime;
        if (width_ != 2 || !col.prefill || col.batch != 1 || col.rope_positions == nullptr ||
            col.rope_positions->numel() != T ||
            (feature_tap_ != nullptr && !feature_tap_->captures_columns()) ||
            (runtime != nullptr && runtime->taps != nullptr) ||
            prefill_slices_mode_env() == PrefillSlicesMode::Force) {
            return 0;
        }
        const std::int32_t split = prefill_overlap_split(T);
        return split > 0 && ops::allreduce_sum_split_available(*tp_.execution, *tp_.events) ? split
                                                                                            : 0;
    } else {
        (void)T;
        return 0;
    }
}

template <class Arena>
PrefillSlicePlan Qwen4Forward<Arena>::prefill_slices(std::int32_t T) const {
    if constexpr (kExecute) {
        if (width_ != 2 || !c(0).prefill || prefill_slices_mode_env() == PrefillSlicesMode::Off ||
            !ops::allreduce_sum_split_available(*tp_.execution, *tp_.events)) {
            return {};
        }
        return prefill_slice_plan(T, prefill_slice_tokens_env());
    } else {
        (void)T;
        return {};
    }
}

template <class Arena>
void Qwen4Forward<Arena>::finish_slices() {
    if constexpr (kExecute) {
        for (std::int32_t lane = 0; lane < slices_.count; ++lane) {
            ops::allreduce_sum_finish(*tp_.execution, *tp_.events, static_cast<int>(lane));
        }
    }
}

template <class Arena>
auto Qwen4Forward<Arena>::plain_halves(const RankTensors& x, const RankTensors& staging,
                                       std::int32_t split) const -> std::array<PlainHalf, 2> {
    const std::int32_t T      = columns();
    const Qwen4Columns& chunk = c(0);
    if (split <= 0 || split >= T || chunk.envelope == nullptr ||
        chunk.envelope->max_visible_keys != static_cast<std::uint32_t>(chunk.first_position + T)) {
        throw std::logic_error("prefill overlap: invalid half split");
    }
    std::array<PlainHalf, 2> halves;
    for (int h = 0; h < 2; ++h) {
        PlainHalf& half          = halves[at(h)];
        const std::int32_t first = h == 0 ? 0 : split;
        const std::int32_t count = h == 0 ? split : T - split;
        half.first               = first;
        half.x                   = window(x, first, count);
        half.staging             = window(staging, first, count);
        for (int r = 0; r < width_; ++r) {
            half.positions[at(r)] = c(r).cache_positions->slice(0, first, count);
            half.rope[at(r)]      = c(r).rope_positions->slice(0, first, count);
        }
        const auto visible = static_cast<std::uint32_t>(chunk.first_position + first + count);
        half.envelope      = {visible, visible};
    }
    return halves;
}

template <class Arena>
template <class Body>
void Qwen4Forward<Arena>::half_block(PlainHalf& half, int lane, Body&& body, int tapped_layer) {
    finish_half(half, lane);
    const std::array<Qwen4Columns, 2> chunk{r_[0].columns, r_[1].columns};
    const auto restore = [&] {
        for (int r = 0; r < width_; ++r) { r_[at(r)].columns = chunk[at(r)]; }
        lane_ = -1;
    };
    for (int r = 0; r < width_; ++r) {
        Qwen4Columns& col   = r_[at(r)].columns;
        col.width           = half.x[at(r)].ne[1];
        col.cache_positions = &half.positions[at(r)];
        col.rope_positions  = &half.rope[at(r)];
        col.max_visible     = half.envelope.max_visible_keys;
        col.envelope        = &half.envelope;
        col.source_slot =
            half.first == 0 ? chunk[at(r)].source_slot : chunk[at(r)].destination_slot;
        col.first_position = chunk[at(r)].first_position + half.first;
    }
    lane_ = lane;
    try {
        body(half);
    } catch (...) {
        restore();
        throw;
    }
    restore();
    half.open         = true;
    half.tapped_layer = tapped_layer;
}

template <class Arena>
void Qwen4Forward<Arena>::finish_half(PlainHalf& half, int lane) {
    if (!half.open) { return; }
    half.open              = false;
    const int tapped_layer = half.tapped_layer;
    half.tapped_layer      = -1;
    if constexpr (kExecute) {
        ops::allreduce_sum_finish(*tp_.execution, *tp_.events, lane);
        if (feature_tap_ != nullptr && tapped_layer >= 0) {
            feature_tap_->capture_layer_columns(tapped_layer, half.x, half.first);
        }
    }
}

template <class Arena>
void Qwen4Forward<Arena>::abandon_lanes(std::array<PlainHalf, 2>& halves) noexcept {
    if constexpr (kExecute) {
        for (std::int32_t lane = 0; lane < slices_.count; ++lane) {
            try {
                ops::allreduce_sum_finish(*tp_.execution, *tp_.events, static_cast<int>(lane));
            } catch (...) {}
        }
    }
    for (int h = 0; h < 2; ++h) {
        halves[at(h)].tapped_layer = -1;
        try {
            finish_half(halves[at(h)], h);
        } catch (...) {}
    }
}

template <class Arena>
std::array<ExchangePrefetch, 2> Qwen4Forward<Arena>::ffn_input_hint(std::size_t layer,
                                                                    std::size_t bytes) const {
    std::array<ExchangePrefetch, 2> hint;
    for (int r = 0; r < width_; ++r) {
        if (const auto* dense = std::get_if<DenseParameters>(&block(r, layer).ffn)) {
            append_ffn_input_prefetch(*dense, bytes, hint[at(r)]);
        }
    }
    return hint;
}

template <class Arena>
std::array<ExchangePrefetch, 2> Qwen4Forward<Arena>::next_input_hint(std::size_t layer,
                                                                     std::size_t bytes) const {
    std::array<ExchangePrefetch, 2> hint;
    for (int r = 0; r < width_; ++r) {
        ExchangePrefetch& out = hint[at(r)];
        if (layer + 1 >= config_.layer_types.size()) {
            append_prefetch_prefix(parameters(r).text.output_head.weight, bytes, out);
        } else if (const auto* a = std::get_if<AttentionParameters>(&block(r, layer + 1).mixer)) {
            append_mixer_input_prefetch(*a, bytes, out);
        } else if (const auto* g = std::get_if<GdnParameters>(&block(r, layer + 1).mixer)) {
            append_mixer_input_prefetch(*g, bytes, out);
        }
    }
    return hint;
}

template <class Arena>
void Qwen4Forward<Arena>::plain_norm(std::size_t layer, Tensor Qwen4BlockParameters::* member,
                                     const RankTensors& x, RankTensors& h) {
    each([&](int r) {
        if constexpr (kExecute) {
            ops::rmsnorm(x[at(r)], block(r, layer).*member, config_.rms_norm_eps, true, h[at(r)],
                         s(r));
        }
    });
}

} // namespace ninfer::models::qwen3_5::execution
