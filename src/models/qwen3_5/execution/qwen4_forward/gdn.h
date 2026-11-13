#pragma once

// Gated DeltaNet mixer over each rank's key/value heads, conv channels and state slots: the
// gating controls, the input projection with its causal convolution (prompt chunk, decode
// snapshot or ReplaySSM record), the recurrence and the gated output norm.
//
// Member definitions of Qwen4Forward (qwen4_forward.h), included only by qwen4_forward.cpp, which
// instantiates them for execution and for planning.

#include "models/qwen3_5/execution/qwen4_forward.h"
#include "models/qwen3_5/execution/gdn.h"
#include "models/qwen3_5/execution/tp2/gdn_split.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "models/qwen3_5/execution/tp2/gdn_gating_overlap.h"
#include "ninfer/ops/hyper_connection.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <variant>

namespace ninfer::models::qwen3_5::execution {

template <class Arena>
void Qwen4Forward<Arena>::gdn_join_gating() {
    if constexpr (kExecute) {
        if (gating_fork_ == nullptr) { return; }
        for (int r = 0; r < width_; ++r) {
            const ScopedCurrentDevice scope(r_[at(r)].device);
            CUDA_CHECK(cudaStreamWaitEvent(s(r), gating_fork_->join[at(r)], 0));
        }
        gating_fork_ = nullptr;
    }
}

// The one parent of a GDN projection that the generic chain runs (a paired projection has fused
// routes only).
[[nodiscard]] inline const LinearParameters& generic_projection(const ops::ProjectionWeights& w) {
    const auto* one = std::get_if<LinearParameters>(&w);
    if (one == nullptr) {
        throw std::invalid_argument("GDN: a paired projection has no generic route");
    }
    return *one;
}

template <class Arena>
void Qwen4Forward<Arena>::gdn(std::size_t layer, const RankTensors& in, RankTensors& y,
                              std::int32_t gidx, const RankTensors& staging) {
    const std::int32_t T = columns();
    const auto& g        = *rc().gdn;
    const auto heads     = dimension(g.linear_num_value_heads);
    const auto dv        = dimension(g.linear_value_head_dim);
    const auto kw        = dimension(g.key_width());
    const auto vw        = dimension(g.value_width());
    const bool plain     = !has_hc();
    const auto gl        = static_cast<std::uint32_t>(gidx);
    const auto p         = [&](int r) -> const GdnParameters& {
        return std::get<GdnParameters>(block(r, layer).mixer);
    };
    auto scope = scopes();
    RankTensors h;
    if (plain) { h = alloc(DType::BF16, {hidden(), T}); }
    const RankTensors& x = plain ? h : in;
    RankTensors decay    = alloc(DType::FP32, {heads, T});
    RankTensors beta     = alloc(DType::FP32, {heads, T});
    RankTensors z        = alloc(DType::BF16, {vw, T});
    RankTensors qc       = alloc(DType::BF16, {kw, T});
    RankTensors kc       = alloc(DType::BF16, {kw, T});
    RankTensors vc       = alloc(DType::BF16, {vw, T});
    RankTensors o        = alloc(DType::BF16, {dv, heads, T});
    RankTensors on       = alloc(DType::BF16, {dv, heads, T});
    // The z the output norm reads: `z`, or (generic, one column) a view of the projection.
    RankTensors zs = z;

    gdn_controls(layer, in, h, decay, beta);

    // The input projection and the causal convolution (q, k, v; z bypasses it).
    if (block(0, layer).fused.gdn_input) {
        fused_gdn_input(layer, x, gl, gidx, z, qc, kc, vc);
    } else {
        generic_gdn_input(layer, x, gl, gidx, z, zs, qc, kc, vc);
    }

    gdn_join_gating();
    gdn_recurrence(layer, gl, gidx, qc, kc, vc, decay, beta, zs, o, on);
    const RankTensors flat{on[0].view({vw, T}), width_ == 2 ? on[1].view({vw, T}) : Tensor{}};
    close(flat, {&p(0).output, width_ == 2 ? &p(1).output : nullptr}, y, staging,
          static_cast<std::int32_t>(layer), 0);
}

template <class Arena>
void Qwen4Forward<Arena>::gdn_controls(std::size_t layer, const RankTensors& in, RankTensors& h,
                                       RankTensors& decay, RankTensors& beta) {
    const std::int32_t T          = columns();
    const auto& g                 = *rc().gdn;
    const auto heads              = dimension(g.linear_num_value_heads);
    const Qwen4FusedRoutes& fused = block(0, layer).fused;
    const bool plain              = !has_hc();
    const RankTensors& x          = plain ? h : in;
    const auto p                  = [&](int r) -> const GdnParameters& {
        return std::get<GdnParameters>(block(r, layer).mixer);
    };
    if (plain && width_ == 1 && fused.gdn_gating) {
        if constexpr (kExecute) {
            gdn_norm_control(in[0], block(0, layer).input_norm, config_.rms_norm_eps, p(0), h[0],
                             decay[0], beta[0], work(0), r_[0].execution);
        } else {
            reserve(
                ops::gdn_norm_gating_proj_workspace_capacity_bytes(heads, hidden(), first_, last_));
        }
    } else {
        if (plain) { plain_norm(layer, &Qwen4BlockParameters::input_norm, in, h); }
        if (fused.gdn_gating) {
            if constexpr (kExecute) {
                if (width_ == 1) {
                    auto call = work(0).scope();
                    ops::gdn_gating_proj(x[0], generic_projection(p(0).control).weight, p(0).a_log,
                                         p(0).dt_bias, work(0), decay[0], beta[0], r_[0].execution);
                } else {
                    const std::array<WorkspaceArena*, 2> ws{&work(0), &work(1)};
                    if (plain && T <= 32 && tp_.gating != nullptr) {
                        GdnGatingOverlap& overlap = *tp_.gating;
                        for (std::size_t r = 0; r < 2; ++r) {
                            const ScopedCurrentDevice scope(tp_.execution->dev[r]->device);
                            CUDA_CHECK(cudaEventRecord(overlap.fork[r], s(static_cast<int>(r))));
                            CUDA_CHECK(cudaStreamWaitEvent(overlap.side.dev[r]->stream,
                                                           overlap.fork[r], 0));
                        }
                        gdn_control_split(x, {&p(0), &p(1)}, decay, beta, ws, overlap.side);
                        for (std::size_t r = 0; r < 2; ++r) {
                            const ScopedCurrentDevice scope(tp_.execution->dev[r]->device);
                            CUDA_CHECK(
                                cudaEventRecord(overlap.join[r], overlap.side.dev[r]->stream));
                        }
                        gating_fork_ = &overlap;
                    } else {
                        gdn_control_split(x, {&p(0), &p(1)}, decay, beta, ws, *tp_.execution);
                    }
                }
            } else {
                reserve(width_ == 1
                            ? ops::gdn_gating_proj_workspace_capacity_bytes(heads, hidden(), first_,
                                                                            last_)
                            : gdn_control_split_workspace_bytes(g, hidden(), first_, last_));
            }
        } else {
            auto control_scope  = scopes();
            RankTensors control = alloc(DType::BF16, {2 * heads, T});
            each([&](int r) {
                linear(r, x[at(r)], generic_projection(p(r).control), control[at(r)]);
                if constexpr (kExecute) {
                    // a|b read in place from the projection (T49: no row-split copies).
                    ops::gdn_gating(control[at(r)].slice(0, 0, heads),
                                    control[at(r)].slice(0, heads, heads), p(r).a_log, p(r).dt_bias,
                                    decay[at(r)], beta[at(r)], s(r));
                }
            });
        }
    }
}

template <class Arena>
void Qwen4Forward<Arena>::generic_gdn_input(std::size_t layer, const RankTensors& x,
                                            std::uint32_t gl, std::int32_t gidx,
                                            const RankTensors& z, RankTensors& zs, RankTensors& qc,
                                            RankTensors& kc, RankTensors& vc) {
    const std::int32_t T = columns();
    const auto& g        = *rc().gdn;
    const auto kw        = dimension(g.key_width());
    const auto vw        = dimension(g.value_width());
    const auto conv      = dimension(g.conv_channels());
    const auto p         = [&](int r) -> const GdnParameters& {
        return std::get<GdnParameters>(block(r, layer).mixer);
    };
    RankTensors projection = alloc(DType::BF16, {conv + vw, T});
    RankTensors qkv        = alloc(DType::BF16, {conv, T});
    each([&](int r) {
        linear(r, x[at(r)], generic_projection(p(r).projection), projection[at(r)]);
    });
    if constexpr (kExecute) {
        each([&](int r) {
            const std::size_t i            = at(r);
            const GdnParameters& gp        = p(r);
            const Qwen4Columns& col        = c(r);
            LinearAttentionStatePool& pool = *state(r).linear;
            cudaStream_t stream            = s(r);
            zs[i]                          = split_rows(r, projection[i], conv, vw, z[i]);
            if (col.prefill) {
                const Tensor qkvs = split_rows(r, projection[i], 0, conv, qkv[i]);
                Tensor conv_in    = pool.conv_slot(gl, col.source_slot);
                Tensor conv_out   = pool.conv_slot(gl, col.destination_slot);
                ops::causal_conv1d_silu_split(qkvs, gp.convolution, conv_in, conv_out, qc[i], kc[i],
                                              vc[i], stream);
                return;
            }
            const std::int32_t W = col.width, B = col.batch;
            Tensor conv_states = pool.layer_view(gl).conv;
            const Tensor valid = col.valid_columns != nullptr ? *col.valid_columns : Tensor{};
            Tensor q3          = qc[i].view({kw, W, B});
            Tensor k3          = kc[i].view({kw, W, B});
            Tensor v3          = vc[i].view({vw, W, B});
            if (col.replay_records != nullptr) {
                // ReplaySSM verification: the projected conv inputs are the record itself.
                GdnReplayRecordLayer records = col.replay_records->layer(gidx, B);
                if (records.conv.ne[1] != W) {
                    throw std::logic_error("Text forward ReplaySSM record width is not the window");
                }
                copy_rows(r, projection[i], 0, conv, records.conv.view({conv, T}));
                ops::gdn_projected_conv_record(records.conv, gp.convolution, conv_states, valid,
                                               *col.source_slots, q3, k3, v3, stream);
            } else {
                const Tensor qkvs = split_rows(r, projection[i], 0, conv, qkv[i]);
                ops::gdn_projected_conv_snapshot(qkvs.view({conv, W, B}), gp.convolution,
                                                 conv_states, valid, *col.source_slots,
                                                 *col.destination_slots, q3, k3, v3, stream);
            }
        });
    }
}

template <class Arena>
void Qwen4Forward<Arena>::gdn_recurrence(std::size_t layer, std::uint32_t gl, std::int32_t gidx,
                                         RankTensors& qc, RankTensors& kc, RankTensors& vc,
                                         RankTensors& decay, RankTensors& beta,
                                         const RankTensors& zs, RankTensors& o, RankTensors& on) {
    const std::int32_t T = columns();
    const auto& g        = *rc().gdn;
    const auto heads     = dimension(g.linear_num_value_heads);
    const auto kheads    = dimension(g.linear_num_key_heads);
    const auto dk        = dimension(g.linear_key_head_dim);
    const auto dv        = dimension(g.linear_value_head_dim);
    const auto p         = [&](int r) -> const GdnParameters& {
        return std::get<GdnParameters>(block(r, layer).mixer);
    };
    if constexpr (!kExecute) {
        if (c(0).prefill) {
            reserve(ops::gated_delta_net_workspace_capacity_bytes(kheads, heads, first_, last_));
        }
    } else {
        each([&](int r) {
            const std::size_t i            = at(r);
            const Qwen4Columns& col        = c(r);
            LinearAttentionStatePool& pool = *state(r).linear;
            cudaStream_t stream            = s(r);
            const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(dk)));
            if (col.prefill) {
                Tensor state_in  = pool.recurrent_slot(gl, col.source_slot);
                Tensor state_out = pool.recurrent_slot(gl, col.destination_slot);
                ops::gated_delta_net(qc[i].view({dk, kheads, T}), kc[i].view({dk, kheads, T}),
                                     vc[i].view({dv, heads, T}), decay[i], beta[i], scale,
                                     /*normalize_qk=*/true, work(r), state_in, state_out, o[i],
                                     r_[i].execution);
            } else {
                const std::int32_t W = col.width, B = col.batch;
                const Tensor valid = col.valid_columns != nullptr ? *col.valid_columns : Tensor{};
                Tensor recurrent   = pool.layer_view(gl).recurrent;
                Tensor o4          = o[i].view({dv, heads, W, B});
                if (col.replay_records != nullptr) {
                    GdnReplayRecordLayer records = col.replay_records->layer(gidx, B);
                    ops::gated_delta_net_replay_record(
                        qc[i].view({dk, kheads, W, B}), kc[i].view({dk, kheads, W, B}),
                        vc[i].view({dv, heads, W, B}), decay[i].view({heads, W, B}),
                        beta[i].view({heads, W, B}), scale, recurrent, valid, *col.source_slots,
                        records.key, records.value, records.gate, o4, stream);
                } else {
                    ops::gated_delta_net_batch_update(
                        qc[i].view({dk, kheads, W, B}), kc[i].view({dk, kheads, W, B}),
                        vc[i].view({dv, heads, W, B}), decay[i].view({heads, W, B}),
                        beta[i].view({heads, W, B}), scale, /*normalize_qk=*/true, recurrent,
                        *col.source_slots, *col.destination_slots, o4, stream);
                }
            }
            // output_gate_type: sigmoid (Qwen3.8-Flash-Next) or SiLU (Qwen3.5).
            if (config_.gdn_gate == GateActivation::Sigmoid) {
                ops::gated_rmsnorm_sigmoid(o[i], p(r).norm, zs[i].view({dv, heads, T}),
                                           config_.rms_norm_eps, on[i], stream);
            } else {
                ops::gated_rmsnorm(o[i], p(r).norm, zs[i].view({dv, heads, T}),
                                   config_.rms_norm_eps, on[i], stream);
            }
        });
    }
}

template <class Arena>
void Qwen4Forward<Arena>::fused_gdn_input(std::size_t layer, const RankTensors& x, std::uint32_t gl,
                                          std::int32_t gidx, RankTensors& z, RankTensors& qc,
                                          RankTensors& kc, RankTensors& vc) {
    const std::int32_t T = columns();
    const auto conv      = dimension(rc().gdn->conv_channels());
    const auto p         = [&](int r) -> const GdnParameters& {
        return std::get<GdnParameters>(block(r, layer).mixer);
    };
    const Qwen4Columns& col0 = c(0);
    if (col0.prefill) {
        auto conv_scope = scopes();
        RankTensors qkv = alloc(DType::BF16, {conv, T});
        if constexpr (kExecute) {
            if (width_ == 1) {
                gdn_projection(x[0], p(0), qkv[0], z[0], work(0), s(0));
            } else {
                const std::array<WorkspaceArena*, 2> ws{&work(0), &work(1)};
                gdn_projection_split(x, {&p(0), &p(1)}, qkv, z, ws, *tp_.execution);
            }
            each([&](int r) {
                LinearAttentionStatePool& pool = *state(r).linear;
                Tensor conv_in                 = pool.conv_slot(gl, c(r).source_slot);
                Tensor conv_out                = pool.conv_slot(gl, c(r).destination_slot);
                ops::causal_conv1d_silu_split(qkv[at(r)], p(r).convolution, conv_in, conv_out,
                                              qc[at(r)], kc[at(r)], vc[at(r)], s(r));
            });
        } else {
            for (int r = 0; r < width_; ++r) {
                reserve(width_ == 1 ? gdn_projection_workspace_bytes(p(r), first_, last_)
                                    : gdn_projection_split_workspace_bytes(p(r), first_, last_),
                        r);
            }
        }
        return;
    }
    fused_gdn_window(layer, x, gl, gidx, z, qc, kc, vc);
}

template <class Arena>
void Qwen4Forward<Arena>::fused_gdn_window(std::size_t layer, const RankTensors& x,
                                           std::uint32_t gl, std::int32_t gidx, RankTensors& z,
                                           RankTensors& qc, RankTensors& kc, RankTensors& vc) {
    const auto& g = *rc().gdn;
    const auto kw = dimension(g.key_width());
    const auto vw = dimension(g.value_width());
    const auto p  = [&](int r) -> const GdnParameters& {
        return std::get<GdnParameters>(block(r, layer).mixer);
    };
    const Qwen4Columns& col0 = c(0);
    const std::int32_t W = col0.width, B = col0.batch;
    if constexpr (!kExecute) {
        // Planning does not know whether the window records (verification) or snapshots.
        for (int r = 0; r < width_; ++r) {
            std::size_t bytes = width_ == 1 ? gdn_snapshot_workspace_bytes(p(r), g, B, W, W)
                                            : gdn_snapshot_split_workspace_bytes(p(r), B, W, W);
            if (W >= 2) {
                bytes =
                    std::max(bytes, width_ == 1 ? gdn_record_workspace_bytes(p(r), g, B, W, W)
                                                : gdn_record_split_workspace_bytes(p(r), B, W, W));
            }
            reserve(bytes, r);
        }
        (void)gidx;
    } else {
        RankTensors h3, q3, k3, v3, z3, conv_states, valid, sources, destinations, records;
        for (int r = 0; r < width_; ++r) {
            const std::size_t i     = at(r);
            const Qwen4Columns& col = c(r);
            h3[i]                   = x[i].view({hidden(), W, B});
            q3[i]                   = qc[i].view({kw, W, B});
            k3[i]                   = kc[i].view({kw, W, B});
            v3[i]                   = vc[i].view({vw, W, B});
            z3[i]                   = z[i].view({vw, W, B});
            conv_states[i]          = state(r).linear->layer_view(gl).conv;
            valid[i]                = col.valid_columns != nullptr ? *col.valid_columns : Tensor{};
            sources[i]              = *col.source_slots;
            if (col.replay_records != nullptr) {
                records[i] = col.replay_records->layer(gidx, B).conv;
                if (records[i].ne[1] != W) {
                    throw std::logic_error("ReplaySSM record width is not the window");
                }
            } else {
                destinations[i] = *col.destination_slots;
            }
        }
        if (col0.replay_records != nullptr) {
            if (width_ == 1) {
                gdn_projection_record(h3[0], p(0), g, conv_states[0], valid[0], sources[0],
                                      records[0], q3[0], k3[0], v3[0], z3[0], work(0), s(0));
            } else {
                const std::array<WorkspaceArena*, 2> ws{&work(0), &work(1)};
                gdn_projection_record_split(h3, {&p(0), &p(1)}, conv_states, valid, sources,
                                            records, q3, k3, v3, z3, ws, *tp_.execution);
            }
        } else {
            if (width_ == 1) {
                gdn_projection_snapshot(h3[0], p(0), g, conv_states[0], valid[0], sources[0],
                                        destinations[0], q3[0], k3[0], v3[0], z3[0], work(0), s(0));
            } else {
                const std::array<WorkspaceArena*, 2> ws{&work(0), &work(1)};
                gdn_projection_snapshot_split(h3, {&p(0), &p(1)}, conv_states, valid, sources,
                                              destinations, q3, k3, v3, z3, ws, *tp_.execution);
            }
        }
    }
}

} // namespace ninfer::models::qwen3_5::execution
