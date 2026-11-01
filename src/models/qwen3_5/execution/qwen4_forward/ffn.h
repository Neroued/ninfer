#pragma once

// The FFN of a block: the dispatch between the MoE and the dense SwiGLU FFN of a hyper-connection
// block, and the dense FFN with its fused or generic gate|up projection, also as the plain
// block's FFN with the tp 2 prompt slices.
//
// Member definitions of Qwen4Forward (qwen4_forward.h), included only by qwen4_forward.cpp, which
// instantiates them for execution and for planning.

#include "models/qwen3_5/execution/qwen4_forward.h"
#include "models/qwen3_5/execution/ffn.h"
#include "models/qwen3_5/execution/tp2/ffn_split.h"
#include "models/qwen3_5/execution/tp2/linear_split.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/silu_mul.h"

#include <variant>

namespace ninfer::models::qwen3_5::execution {

template <class Arena>
void Qwen4Forward<Arena>::plain_ffn(std::size_t layer, RankTensors& x, const RankTensors& staging) {
    const std::int32_t T = columns();
    const auto p         = [&](int r) -> const DenseParameters& {
        return std::get<DenseParameters>(block(r, layer).ffn);
    };
    auto scope    = scopes();
    RankTensors h = alloc(DType::BF16, {hidden(), T});
    if constexpr (kExecute) {
        if (slices_.usable()) {
            // Slice s waits for its mixer all-reduce (lane s) only; its FFN all-reduce begins
            // on the same lane (run_plain finishes the lanes).
            const std::array<WorkspaceArena*, 2> ws{&work(0), &work(1)};
            for (std::int32_t slice = 0; slice < slices_.count; ++slice) {
                const std::int32_t first = slices_.first(slice);
                const std::int32_t count = slices_.columns(slice, T);
                ops::allreduce_sum_finish(*tp_.execution, *tp_.events, static_cast<int>(slice));
                const RankTensors xs = window(x, first, count);
                RankTensors hs       = window(h, first, count);
                plain_norm(layer, &Qwen4BlockParameters::post_attention_norm, xs, hs);
                dense_ffn_split(hs, {&p(0), &p(1)}, xs, window(staging, first, count), ws,
                                *tp_.execution, *tp_.events, static_cast<int>(slice));
            }
            return;
        }
    }
    plain_norm(layer, &Qwen4BlockParameters::post_attention_norm, x, h);
    if constexpr (kExecute) {
        if (width_ == 1) {
            dense_ffn(h[0], p(0), x[0], work(0), s(0));
        } else {
            const std::array<WorkspaceArena*, 2> ws{&work(0), &work(1)};
            dense_ffn_split(h, {&p(0), &p(1)}, x, staging, ws, *tp_.execution, *tp_.events, lane_);
        }
    } else {
        for (int r = 0; r < width_; ++r) {
            reserve(width_ == 1 ? dense_ffn_workspace_bytes(p(r), first_, last_)
                                : dense_ffn_split_workspace_bytes(p(r), first_, last_),
                    r);
        }
    }
}

template <class Arena>
void Qwen4Forward<Arena>::ffn(std::size_t layer, const RankTensors& wide, const RankTensors& x,
                              RankTensors& y, const RankTensors& staging_bf16,
                              const RankTensors& staging_fp32) {
    if (moe_ffn()) {
        ahead(layer, wide);
        moe(layer, x, y, staging_fp32);
    } else {
        dense(layer, x, y, staging_bf16);
    }
}

template <class Arena>
void Qwen4Forward<Arena>::dense(std::size_t layer, const RankTensors& x, RankTensors& y,
                                const RankTensors& staging) {
    const std::int32_t T    = columns();
    const auto intermediate = dimension(std::get<DenseConfig>(rc().ffn).intermediate_size);
    const auto p            = [&](int r) -> const DenseParameters& {
        return std::get<DenseParameters>(block(r, layer).ffn);
    };
    auto scope             = scopes();
    RankTensors activation = alloc(DType::BF16, {intermediate, T});
    if (block(0, layer).fused.swiglu) {
        if constexpr (kExecute) {
            if (width_ == 1) {
                auto call = work(0).scope();
                ops::linear_swiglu(x[0], p(0).gate_up.weight, activation[0], p(0).gate_up.policy,
                                   work(0), s(0));
            } else {
                const std::array<WorkspaceArena*, 2> ws{&work(0), &work(1)};
                project_swiglu_column_parallel(x, {&p(0).gate_up, &p(1).gate_up}, activation, ws,
                                               *tp_.execution);
            }
        } else {
            for (int r = 0; r < width_; ++r) {
                const LinearParameters& gu = p(r).gate_up;
                reserve(ops::linear_swiglu_workspace_capacity_bytes(
                            gu.weight.qtype, gu.weight.n, gu.weight.k, gu.policy, first_, last_),
                        r);
            }
        }
    } else {
        auto gate_scope     = scopes();
        RankTensors gate_up = alloc(DType::BF16, {2 * intermediate, T});
        each([&](int r) {
            const std::size_t i = at(r);
            linear(r, x[i], p(r).gate_up, gate_up[i]);
            if constexpr (kExecute) {
                ops::silu_mul(gate_up[i].slice(0, 0, intermediate),
                              gate_up[i].slice(0, intermediate, intermediate), activation[i], s(r));
            }
        });
    }
    close(activation, {&p(0).down, width_ == 2 ? &p(1).down : nullptr}, y, staging,
          static_cast<std::int32_t>(layer), 1);
}

} // namespace ninfer::models::qwen3_5::execution
