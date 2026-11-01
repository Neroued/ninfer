#pragma once

// Dense gated full attention of a plain block (a model without QSA): q|k|gate|v projection,
// q/k RMSNorm and RoPE (tp 2: the fused norm+RoPE Op), causal attention over the paged KV within
// the call's envelope, sigmoid output gate and the closing projection.
//
// Member definitions of Qwen4Forward (qwen4_forward.h), included only by qwen4_forward.cpp, which
// instantiates them for execution and for planning.

#include "models/qwen3_5/execution/qwen4_forward.h"
#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/tp2/attention_split.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/softmax_attention.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <variant>

namespace ninfer::models::qwen3_5::execution {

template <class Arena>
void Qwen4Forward<Arena>::dense_attention(std::size_t layer, const RankTensors& in, RankTensors& y,
                                          std::int32_t fidx, const RankTensors& staging) {
    const std::int32_t T = columns();
    const auto& a        = *rc().attention;
    const auto head_dim  = dimension(a.head_dim);
    const auto heads     = dimension(a.num_attention_heads);
    const auto kv_heads  = dimension(a.num_key_value_heads);
    const auto qw        = dimension(a.query_width());
    const auto kw        = dimension(a.key_width());
    const auto p         = [&](int r) -> const AttentionParameters& {
        return std::get<AttentionParameters>(block(r, layer).mixer);
    };
    auto scope    = scopes();
    RankTensors x = alloc(DType::BF16, {hidden(), T});
    plain_norm(layer, &Qwen4BlockParameters::input_norm, in, x);
    RankTensors q    = alloc(DType::BF16, {qw, T});
    RankTensors gate = alloc(DType::BF16, {qw, T});
    RankTensors k    = alloc(DType::BF16, {kw, T});
    RankTensors v    = alloc(DType::BF16, {kw, T});
    if constexpr (kExecute) {
        if (width_ == 1) {
            attention_projection(x[0], p(0), q[0], gate[0], k[0], v[0], work(0), s(0));
        } else {
            const std::array<WorkspaceArena*, 2> ws{&work(0), &work(1)};
            attention_projection_split(x, {&p(0), &p(1)}, q, gate, k, v, ws, *tp_.execution);
        }
    } else {
        for (int r = 0; r < width_; ++r) {
            reserve(width_ == 1 ? attention_projection_workspace_bytes(p(r), first_, last_)
                                : attention_projection_split_workspace_bytes(p(r), first_, last_),
                    r);
        }
    }
    RankTensors qn       = alloc(DType::BF16, {head_dim, heads, T});
    RankTensors kn       = alloc(DType::BF16, {head_dim, kv_heads, T});
    RankTensors attended = alloc(DType::BF16, {head_dim, heads, T});
    if constexpr (!kExecute) {
        const std::int32_t W = c(0).width, B = c(0).batch;
        const bool batched = !c(0).prefill;
        const ops::CausalAttentionExecutionEnvelope envelope{1, std::max(c(0).max_visible, 1U)};
        reserve(ops::causal_softmax_attention_workspace_capacity_bytes(
            {head_dim, heads, kv_heads}, c(0).kv_storage, envelope, batched ? B : 1,
            batched ? W : first_, batched ? W : last_));
    } else {
        each([&](int r) { dense_attention_rank(layer, r, fidx, q, gate, k, v, qn, kn, attended); });
    }
    const RankTensors flat{attended[0].view({qw, T}),
                           width_ == 2 ? attended[1].view({qw, T}) : Tensor{}};
    close(flat, {&p(0).output, width_ == 2 ? &p(1).output : nullptr}, y, staging,
          static_cast<std::int32_t>(layer), 0);
}

template <class Arena>
void Qwen4Forward<Arena>::dense_attention_rank(std::size_t layer, int r, std::int32_t fidx,
                                               const RankTensors& q, const RankTensors& gate,
                                               const RankTensors& k, const RankTensors& v,
                                               RankTensors& qn, RankTensors& kn,
                                               RankTensors& attended) {
    if constexpr (kExecute) {
        const std::int32_t T         = columns();
        const auto& a                = *rc().attention;
        const auto head_dim          = dimension(a.head_dim);
        const auto heads             = dimension(a.num_attention_heads);
        const auto kv_heads          = dimension(a.num_key_value_heads);
        const AttentionParameters& p = std::get<AttentionParameters>(block(r, layer).mixer);
        const std::int32_t W = c(0).width, B = c(0).batch;
        const bool batched  = !c(0).prefill;
        const float scale   = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
        const std::size_t i = at(r);
        const Qwen4Columns& col = c(r);
        cudaStream_t stream     = s(r);
        if (col.envelope == nullptr) {
            throw std::logic_error("dense attention has no causal envelope");
        }
        const Tensor q3 = q[i].view({head_dim, heads, T});
        const Tensor k3 = k[i].view({head_dim, kv_heads, T});
        const Tensor v3 = v[i].view({head_dim, kv_heads, T});
        // Decode and verification windows bind [W,B] positions read one-axis; a prompt
        // chunk binds its [T] or [T,3] RoPE positions as they are.
        const Tensor& rope     = *col.rope_positions;
        const Tensor rope_flat = batched ? rope.view({T}) : rope;
        if (width_ == 1) {
            ops::rmsnorm(q3, p.query_norm, config_.rms_norm_eps, true, qn[i], stream);
            ops::rmsnorm(k3, p.key_norm, config_.rms_norm_eps, true, kn[i], stream);
            text_rope(rope_flat, *config_.rope_parameters, qn[i], kn[i], stream);
        } else {
            text_qk_norm_rope(rope_flat, *config_.rope_parameters, p.query_norm, p.key_norm,
                              config_.rms_norm_eps, q3, k3, qn[i], kn[i], stream);
        }
        const PagedKVBatchLayerView cache =
            state(r).kv->batch_layer_view(static_cast<std::uint32_t>(fidx));
        if (batched) {
            const Tensor valid = col.valid_columns != nullptr ? *col.valid_columns : Tensor{};
            Tensor a4          = attended[i].view({head_dim, heads, W, B});
            ops::causal_softmax_attention(
                qn[i].view({head_dim, heads, W, B}), kn[i].view({head_dim, kv_heads, W, B}),
                v[i].view({head_dim, kv_heads, W, B}), col.cache_positions->view({W, B}), valid,
                *col.table_rows, {head_dim, heads, kv_heads}, scale, cache, *col.envelope, work(r),
                a4, stream);
        } else {
            ops::causal_softmax_attention(qn[i], kn[i], v3, *col.cache_positions, Tensor{},
                                          *col.table_rows, {head_dim, heads, kv_heads}, scale,
                                          cache, *col.envelope, work(r), attended[i], stream);
        }
        ops::sigmoid_mul(gate[i].view({head_dim, heads, T}), attended[i], stream);
    } else {
        (void)layer, (void)r, (void)fidx, (void)q, (void)gate, (void)k, (void)v, (void)qn, (void)kn,
            (void)attended;
    }
}

} // namespace ninfer::models::qwen3_5::execution
