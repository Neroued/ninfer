#pragma once

// The hyper-connection residual (docs/maintainer/qwen4-exp-model.md#hyper-connection-residual):
// the embedding widened to S streams, one mixer per block input (the pending block output
// committed in the same pass), the PLE layer, and the output mixer that feeds the head. A model
// without hyper-connections takes the plain residual (plain.h) from run().
//
// Member definitions of Qwen4Forward (qwen4_forward.h), included only by qwen4_forward.cpp, which
// instantiates them for execution and for planning.

#include "models/qwen3_5/execution/qwen4_forward.h"
#include "core/device.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/scatter.h"
#include "ops/sparse_moe/host/host_moe_mailbox.h"

#include <exception>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::execution {

template <class Arena>
void Qwen4Forward<Arena>::run(const Tensor& ids, const Tensor* ple_embedding, Tensor& wide_out,
                              const VisualColumns* visual) {
    const RankTensors rank_ids{ids, Tensor{}};
    RankTensors rank_ple;
    if (ple_embedding != nullptr) { rank_ple = {*ple_embedding, Tensor{}}; }
    RankTensors rank_out{wide_out, Tensor{}};
    run(rank_ids, ple_embedding != nullptr ? &rank_ple : nullptr, rank_out, visual);
}

// The residual state of a hyper-connection forward: the committed wide residual, the next block
// input (`mixed`), the two block outputs and their stream gates (a block writes one pair while the
// next mixer still reads the other), the pending output that the next mixer or a commit folds
// into the residual, and the tp 2 all-reduce staging of the two exchanges per block (FP32 for the
// MoE).
template <class Arena>
struct Qwen4Forward<Arena>::HcStreams {
    RankTensors wide, mixed;
    std::array<RankTensors, 2> y, inject;
    int pending = -1;
    int next    = 0;
    RankTensors staging_bf16, staging_fp32;
};

template <class Arena>
void Qwen4Forward<Arena>::run(const RankTensors& ids, const RankTensors* ple_embedding,
                              RankTensors& wide_out, const VisualColumns* visual) {
    if (!has_hc()) {
        if (ple_embedding != nullptr || config_.ple) {
            throw std::invalid_argument("Qwen4 PLE needs the hyper-connection residual");
        }
        run_plain(ids, wide_out, visual);
        return;
    }
    if (feature_tap_ != nullptr) {
        throw std::invalid_argument("drafter feature taps need the plain residual");
    }
    const std::int32_t T = columns();
    auto scope           = scopes();
    HcStreams st;
    st.wide = wide_buffers(wide_out, T);
    embed_streams(ids, visual, st.wide);
    if (width_ == 2) {
        st.staging_bf16 = alloc(DType::BF16, {hidden(), T});
        if (moe_ffn()) { st.staging_fp32 = alloc(DType::FP32, {hidden(), T}); }
    }
    st.mixed     = alloc(DType::BF16, {hidden(), T});
    st.y[0]      = alloc(DType::BF16, {hidden(), T});
    st.y[1]      = alloc(DType::BF16, {hidden(), T});
    st.inject[0] = alloc(DType::FP32, {streams(), T});
    st.inject[1] = alloc(DType::FP32, {streams(), T});
    for (std::size_t layer = 0; layer < config_.layer_types.size(); ++layer) {
        try {
            hc_block(layer, ple_embedding, st);
        } catch (const std::exception& error) {
            throw std::runtime_error("text/layers/" + std::to_string(layer) + " " + phase() +
                                     (width_ == 2 ? " tp2" : "") + " columns=" + std::to_string(T) +
                                     ": " + error.what());
        }
    }
    hc_commit(st);
    if constexpr (kExecute) {
        if (Qwen4Taps* taps = prompt_taps(-1)) {
            auto tap_scope = scopes();
            RankTensors x  = alloc(DType::BF16, {hidden(), T});
            head(st.wide, x);
            capture_taps(*taps, -1, x);
        }
    }
}

template <class Arena>
void Qwen4Forward<Arena>::embed_streams(const RankTensors& ids, const VisualColumns* visual,
                                        RankTensors& wide) {
    const std::int32_t T = columns();
    auto embed_scope     = scopes();
    RankTensors e        = alloc(DType::BF16, {hidden(), T});
    each([&](int r) {
        if constexpr (kExecute) {
            ops::embedding(ids[at(r)], parameters(r).text.token_embedding, e[at(r)], s(r));
            if (visual != nullptr) {
                ops::scatter(visual->embeddings[at(r)], visual->indices[at(r)], e[at(r)], s(r));
            }
            ops::hyper_connection_expand(e[at(r)], streams(), wide[at(r)], s(r));
        }
    });
}

template <class Arena>
void Qwen4Forward<Arena>::hc_block(std::size_t layer, const RankTensors* ple_embedding,
                                   HcStreams& st) {
    const auto compact   = dimension(config_.compact_layer_indices[layer]);
    const bool attention = config_.layer_types[layer] == MixerKind::FullAttention;
    if (block(0, layer).ple) {
        if (ple_embedding == nullptr) {
            throw std::logic_error("Qwen4Exp PLE block has no n-gram embedding");
        }
        hc_commit(st);
        ple(layer, *ple_embedding, st.wide);
    }
    hc_mix(st, &Qwen4BlockParameters::attention_hc, layer);
    // A hyper-connection model's attention is QSA (qwen4_extensions binds both).
    if (!attention) {
        gdn(layer, st.mixed, st.y[at(st.next)], compact, st.staging_bf16);
    } else {
        qsa(layer, st.mixed, &st.y[at(st.next)], static_cast<std::uint32_t>(compact),
            st.staging_bf16);
    }
    st.pending = st.next;
    st.next ^= 1;
    hc_mix(st, &Qwen4BlockParameters::ffn_hc, layer); // commits the mixer output in the same pass
    ffn(layer, st.wide, st.mixed, st.y[at(st.next)], st.staging_bf16, st.staging_fp32);
    st.pending = st.next;
    st.next ^= 1;
    if constexpr (kExecute) {
        if (Qwen4Taps* taps = prompt_taps(static_cast<std::int32_t>(layer))) {
            hc_commit(st);
            capture_taps(*taps, static_cast<std::int32_t>(layer), st.wide);
        }
    }
}

template <class Arena>
void Qwen4Forward<Arena>::hc_commit(HcStreams& st) {
    if (st.pending < 0) { return; }
    each([&](int r) {
        if constexpr (kExecute) {
            ops::hyper_connection_combine(st.y[at(st.pending)][at(r)],
                                          st.inject[at(st.pending)][at(r)], st.wide[at(r)], s(r));
        }
    });
    st.pending = -1;
}

template <class Arena>
void Qwen4Forward<Arena>::hc_mix(HcStreams& st,
                                 HyperConnectionParameters Qwen4BlockParameters::* member,
                                 std::size_t layer) {
    reserve(ops::hyper_connection_mix_workspace_bytes(first_, last_, true, hidden(), hc_rank()));
    each([&](int r) {
        if constexpr (kExecute) {
            const HyperConnectionParameters& hc = block(r, layer).*member;
            if (st.pending >= 0) {
                ops::hyper_connection_combine_mix(st.wide[at(r)], st.y[at(st.pending)][at(r)],
                                                  st.inject[at(st.pending)][at(r)], hc.weights,
                                                  config_.rms_norm_eps, st.mixed[at(r)],
                                                  &st.inject[at(st.next)][at(r)], work(r), s(r));
            } else {
                ops::hyper_connection_mix(st.wide[at(r)], hc.weights, config_.rms_norm_eps,
                                          st.mixed[at(r)], &st.inject[at(st.next)][at(r)], work(r),
                                          s(r));
            }
        }
    });
    st.pending = -1;
}

template <class Arena>
void Qwen4Forward<Arena>::head(const RankTensors& wide, RankTensors& x) {
    head({&qwen4(0).head, width_ == 2 ? &qwen4(1).head : nullptr}, wide, x);
}

template <class Arena>
void Qwen4Forward<Arena>::head(std::array<const HyperConnectionParameters*, 2> hc,
                               const RankTensors& wide, RankTensors& x) {
    if (!has_hc()) {
        if constexpr (kExecute) {
            for (int r = 0; r < width_; ++r) { x[at(r)] = wide[at(r)]; }
        }
        return;
    }
    reserve(ops::hyper_connection_mix_workspace_bytes(first_, last_, false, hidden(), hc_rank()));
    each([&](int r) {
        if constexpr (kExecute) {
            ops::hyper_connection_mix(wide[at(r)], hc[at(r)]->weights, config_.rms_norm_eps,
                                      x[at(r)], nullptr, work(r), s(r));
        }
    });
}

template <class Arena>
void Qwen4Forward<Arena>::head(const HyperConnectionParameters& hc, const Tensor& wide, Tensor& x) {
    const RankTensors rank_wide{wide, Tensor{}};
    RankTensors rank_x{x, Tensor{}};
    head({&hc, nullptr}, rank_wide, rank_x);
    x = rank_x[0];
}

template <class Arena>
void Qwen4Forward<Arena>::ple(std::size_t layer, const RankTensors& embedding, RankTensors& wide) {
    const std::int32_t T  = columns();
    const auto& ple       = *config_.ple;
    const std::int32_t S  = streams();
    auto scope            = scopes();
    RankTensors key_value = alloc(DType::BF16, {wide_rows() + hidden(), T});
    RankTensors gated     = alloc(DType::BF16, {wide_rows(), T});
    RankTensors normed    = alloc(DType::BF16, {wide_rows(), T});
    RankTensors slots     = alloc(DType::I32, {2});
    each([&](int r) {
        const PleParameters& p = *block(r, layer).ple;
        if constexpr (kExecute) {
            // A decode round's deferred rows: wait for the host's gather, then upload.
            const Qwen4Runtime* runtime = state(r).runtime;
            const Tensor& rows          = embedding[at(r)];
            if (runtime != nullptr && runtime->decode_ple_staged != nullptr &&
                rows.data == runtime->decode_ple_rows.data) {
                ops::host::consume_host_flag(s(r), runtime->decode_ple_staged);
                CUDA_CHECK(cudaMemcpyAsync(rows.data, runtime->decode_ple_host, rows.bytes(),
                                           cudaMemcpyHostToDevice, s(r)));
            }
        }
        linear(r, embedding[at(r)], p.key_value, key_value[at(r)]);
        if constexpr (kExecute) {
            const Qwen4Columns& col = c(r);
            const auto W = col.width, B = col.batch;
            ops::ple_gate(wide[at(r)], key_value[at(r)], p.query_norm, p.key_norm, S,
                          config_.rms_norm_eps, gated[at(r)], s(r));
            ops::grouped_offset_rmsnorm(gated[at(r)], p.conv_norm, S, config_.rms_norm_eps,
                                        normed[at(r)], s(r));
            Tensor states = state(r).runtime->ple_states;
            Tensor source, destination;
            if (col.prefill) {
                source      = slots[at(r)].slice(0, 0, 1);
                destination = slots[at(r)].slice(0, 1, 1);
                ops::set_i32_scalar(source, col.source_slot, s(r));
                ops::set_i32_scalar(destination, col.destination_slot, s(r));
            } else {
                source      = *col.source_slots;
                destination = *col.destination_slots;
            }
            Tensor normed3 = normed[at(r)].view({wide_rows(), W, B});
            Tensor gated3  = gated[at(r)].view({wide_rows(), W, B});
            Tensor wide3   = wide[at(r)].view({wide_rows(), W, B});
            ops::ple_dilated_conv(normed3, gated3, p.convolution,
                                  static_cast<std::int32_t>(ple.conv_dilation()), states, source,
                                  wide3, s(r));
            if (col.replay_records != nullptr) {
                // Verification: the history advances at commit by the accepted prefix.
                const Tensor& record = state(r).runtime->ple_record;
                if (record.data == nullptr || record.ne[0] != wide_rows() || record.ne[1] != W ||
                    record.ne[2] < B) {
                    throw std::logic_error("Qwen4Exp PLE speculative record is unavailable");
                }
                CUDA_CHECK(::ninfer::memcpy_device_async(record.data, normed[at(r)].data,
                                                         normed[at(r)].bytes(), s(r)));
            } else {
                ops::ple_conv_advance(normed3, col.prefill ? nullptr : col.valid_columns,
                                      static_cast<std::int32_t>(ple.conv_history()), states, source,
                                      destination, s(r));
            }
        }
    });
}

} // namespace ninfer::models::qwen3_5::execution
