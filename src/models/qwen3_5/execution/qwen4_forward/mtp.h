#pragma once

// The Qwen4 MTP layer (docs/maintainer/qwen4-exp-model.md#mtp): the stem over the target's wide
// residual and the shifted token, then one QSA block with the MTP experts over the MTP KV cache.
//
// Member definitions of Qwen4Forward (qwen4_forward.h), included only by qwen4_forward.cpp, which
// instantiates them for execution and for planning.

#include "models/qwen3_5/execution/qwen4_forward.h"
#include "core/device.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"

#include <exception>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::execution {

template <class Arena>
void Qwen4Forward<Arena>::run_mtp(std::array<const Qwen4MtpParameters*, 2> m,
                                  const RankTensors& ids, const RankTensors* embeddings,
                                  const RankTensors& hidden_wide, RankTensors& wide_out,
                                  bool append_only) {
    const std::int32_t T = columns();
    if (!has_hc()) {
        throw std::logic_error("Qwen4 MTP layer needs the hyper-connection residual (a model "
                               "without one keeps its Qwen3.5 MTP head)");
    }
    for (int r = 0; r < width_; ++r) {
        if (m[at(r)] == nullptr) {
            throw std::invalid_argument("Qwen4Exp MTP forward rank " + std::to_string(r) +
                                        " has no MTP Parameters");
        }
    }
    mtp_ = m;
    require_ffn({&m[0]->layer.ffn, width_ == 2 ? &m[1]->layer.ffn : nullptr}, "mtp/layers/0");
    auto scope       = scopes();
    RankTensors wide = wide_buffers(wide_out, T);
    mtp_stem(m, ids, embeddings, hidden_wide, wide);
    const std::size_t layer = mtp_layer();
    RankTensors staging_bf16;
    RankTensors staging_fp32;
    if (width_ == 2) {
        staging_bf16 = alloc(DType::BF16, {hidden(), T});
        if (moe_ffn()) { staging_fp32 = alloc(DType::FP32, {hidden(), T}); }
    }
    RankTensors mixed = alloc(DType::BF16, {hidden(), T});
    RankTensors y     = alloc(DType::BF16, {hidden(), T});
    // The attention mixer's gates and the FFN mixer's: combine_mix reads the first while it
    // writes the second, so they are two buffers, as the Text blocks alternate theirs.
    RankTensors attention_inject = alloc(DType::FP32, {streams(), T});
    RankTensors ffn_inject       = alloc(DType::FP32, {streams(), T});
    try {
        reserve(
            ops::hyper_connection_mix_workspace_bytes(first_, last_, true, hidden(), hc_rank()));
        each([&](int r) {
            if constexpr (kExecute) {
                ops::hyper_connection_mix(wide[at(r)], block(r, layer).attention_hc.weights,
                                          config_.rms_norm_eps, mixed[at(r)],
                                          &attention_inject[at(r)], work(r), s(r));
            }
        });
        qsa(layer, mixed, append_only ? nullptr : &y, 0, staging_bf16);
        if (append_only) { return; }
        each([&](int r) {
            if constexpr (kExecute) {
                ops::hyper_connection_combine_mix(
                    wide[at(r)], y[at(r)], attention_inject[at(r)], block(r, layer).ffn_hc.weights,
                    config_.rms_norm_eps, mixed[at(r)], &ffn_inject[at(r)], work(r), s(r));
            }
        });
        if (moe_ffn()) {
            moe(layer, mixed, y, staging_fp32);
        } else {
            dense(layer, mixed, y, staging_bf16);
        }
        each([&](int r) {
            if constexpr (kExecute) {
                ops::hyper_connection_combine(y[at(r)], ffn_inject[at(r)], wide[at(r)], s(r));
            }
        });
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("mtp/layers/0 ") + phase() +
                                 (width_ == 2 ? " tp2" : "") + " columns=" + std::to_string(T) +
                                 ": " + error.what());
    }
}

template <class Arena>
void Qwen4Forward<Arena>::mtp_stem(std::array<const Qwen4MtpParameters*, 2> m,
                                   const RankTensors& ids, const RankTensors* embeddings,
                                   const RankTensors& hidden_wide, RankTensors& wide) {
    const std::int32_t T   = columns();
    auto stem_scope        = scopes();
    RankTensors e          = alloc(DType::BF16, {hidden(), T});
    RankTensors en         = alloc(DType::BF16, {hidden(), T});
    RankTensors projected  = alloc(DType::BF16, {hidden(), T});
    RankTensors hn         = alloc(DType::BF16, {wide_rows(), T});
    RankTensors streams_hp = alloc(DType::BF16, {wide_rows(), T});
    each([&](int r) {
        const std::size_t i         = at(r);
        const Qwen4MtpParameters& p = *m[i];
        if constexpr (kExecute) {
            if (embeddings != nullptr) {
                CUDA_CHECK(::ninfer::memcpy_device_async(e[i].data, (*embeddings)[i].data,
                                                         e[i].bytes(), s(r)));
            } else {
                ops::embedding(ids[i], parameters(r).text.token_embedding, e[i], s(r));
            }
            ops::rmsnorm(e[i], p.embedding_norm, config_.rms_norm_eps, true, en[i], s(r));
            ops::rmsnorm(hidden_wide[i], p.hidden_norm, config_.rms_norm_eps, true, hn[i], s(r));
        }
        linear(r, en[i], p.embedding_projection, projected[i]);
        // The [H,H] projection applies to every stream: [S*H, T] is [H, S*T] column-wise.
        if constexpr (kExecute) {
            const Tensor hn_streams = hn[i].view({hidden(), streams() * T});
            Tensor out_streams      = streams_hp[i].view({hidden(), streams() * T});
            ops::linear(hn_streams, p.hidden_projection.weight, out_streams,
                        p.hidden_projection.policy, work(r), s(r));
            ops::hyper_connection_expand(projected[i], streams(), wide[i], s(r));
            ops::residual_add(streams_hp[i], wide[i], s(r));
        } else {
            const LinearParameters& hp = p.hidden_projection;
            reserve(ops::linear_workspace_capacity_bytes(hp.weight.qtype, hp.weight.n, hp.weight.k,
                                                         hp.policy, first_ * streams(),
                                                         last_ * streams()),
                    r);
        }
    });
}

template <class Arena>
void Qwen4Forward<Arena>::run_mtp(const Qwen4MtpParameters& m, const Tensor& ids,
                                  const Tensor* embeddings, const Tensor& hidden_wide,
                                  Tensor& wide_out, bool append_only) {
    const RankTensors rank_ids{ids, Tensor{}};
    RankTensors rank_embeddings;
    if (embeddings != nullptr) { rank_embeddings = {*embeddings, Tensor{}}; }
    const RankTensors rank_hidden{hidden_wide, Tensor{}};
    RankTensors rank_out{wide_out, Tensor{}};
    run_mtp({&m, nullptr}, rank_ids, embeddings != nullptr ? &rank_embeddings : nullptr,
            rank_hidden, rank_out, append_only);
}

} // namespace ninfer::models::qwen3_5::execution
