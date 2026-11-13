#pragma once

// The routed-expert MoE (docs/maintainer/qwen4-exp-model.md, Execution: MoE): the router, the
// shared expert, the routed experts of the rank's backend into an FP32 partial, and the one BF16
// rounding - at tp 2 after the FP32 all-reduce or the pipelined host exchange of a prompt chunk.
//
// Member definitions of Qwen4Forward (qwen4_forward.h), included only by qwen4_forward.cpp, which
// instantiates them for execution and for planning.

#include "models/qwen3_5/execution/qwen4_forward.h"
#include "models/qwen3_5/execution/qwen4_forward/projection.h"
#include "models/qwen3_5/execution/qwen4_experts.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/moe_grouped.h"
#include "ninfer/ops/moe_route.h"
#include "ninfer/ops/silu_mul.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <variant>

namespace ninfer::models::qwen3_5::execution {

// Publishes a routed-expert call's column groups (null: none) to qwen4_column_groups for the
// duration of the call, and clears it however the call ends.
class ColumnGroupsScope {
public:
    explicit ColumnGroupsScope(Qwen4ColumnGroups* groups) noexcept { qwen4_column_groups = groups; }

    ~ColumnGroupsScope() { qwen4_column_groups = nullptr; }

    ColumnGroupsScope(const ColumnGroupsScope&)            = delete;
    ColumnGroupsScope& operator=(const ColumnGroupsScope&) = delete;
};

template <class Arena>
void Qwen4Forward<Arena>::ahead(std::size_t layer, const RankTensors& wide) {
    if constexpr (kExecute) {
        if (width_ != 1 || r_[0].columns.prefill || layer + 1 >= config_.layer_types.size()) {
            return;
        }
        const Qwen4BlockParameters& next = block(0, layer + 1);
        routed_experts(0, layer)->ahead(static_cast<std::int32_t>(layer), wide[0],
                                        next.ffn_hc.weights.norm, next.moe().router,
                                        config_.rms_norm_eps, s(0));
    } else {
        (void)layer, (void)wide;
    }
}

// The operands of one MoE call on every rank: the routing (top-k ids and weights, the shared
// expert's gate), the shared expert's activations and output, and the FP32 partial of the routed
// experts.
template <class Arena>
struct Qwen4Forward<Arena>::MoeBuffers {
    RankTensors ids, weights, gate, gate_up, activation, shared, partial;
};

template <class Arena>
void Qwen4Forward<Arena>::moe(std::size_t layer, const RankTensors& x, RankTensors& y,
                              const RankTensors& staging) {
    const std::int32_t T = columns();
    const auto shared_w =
        dimension(std::get<MoeConfig>(config_.ffn).shared_expert_intermediate_size);
    auto scope = scopes();
    MoeBuffers b;
    b.ids        = alloc(DType::I32, {ops::kMoeRouteTopK, T});
    b.weights    = alloc(DType::FP32, {ops::kMoeRouteTopK, T});
    b.gate       = alloc(DType::FP32, {T});
    b.gate_up    = alloc(DType::BF16, {2 * shared_w, T});
    b.activation = alloc(DType::BF16, {shared_w, T});
    b.shared     = alloc(DType::BF16, {hidden(), T});
    b.partial    = alloc(DType::FP32, {hidden(), T});
    reserve(ops::moe_route_workspace_bytes(T));
    moe_route_shared(layer, x, b);
    reserve(Qwen4RoutedExperts::workspace_bytes(T));
    if constexpr (kExecute) { moe_combine(layer, x, b, y, staging); }
}

template <class Arena>
void Qwen4Forward<Arena>::moe_route_shared(std::size_t layer, const RankTensors& x, MoeBuffers& b) {
    const std::int32_t T = columns();
    const auto shared_w =
        dimension(std::get<MoeConfig>(config_.ffn).shared_expert_intermediate_size);
    const auto layer_index = static_cast<std::int32_t>(layer);
    each([&](int r) {
        const std::size_t i         = at(r);
        const Qwen4MoeParameters& p = block(r, layer).moe();
        if constexpr (kExecute) {
            // Verify / MTP columns past a lane's valid count route nothing (never committed:
            // no CPU expert work, no cache-policy accesses). The MTP alignment routes only
            // the accepted column (T52): the other columns' outputs are never read.
            const Qwen4Columns& col = c(r);
            const Tensor* valid     = col.prefill ? nullptr : col.valid_columns;
            ops::moe_route(x[i], p.router, work(r), b.ids[i], b.weights[i], b.gate[i], s(r), valid,
                           valid != nullptr && col.moe_last_valid_only
                               ? ops::MoeRouteColumns::LastValid
                               : ops::MoeRouteColumns::ValidPrefix);
            // The backend may start the routed experts elsewhere (the CPU) right away; what
            // it allocates lives until this scope ends.
            routed_experts(r, layer)->routed(layer_index, expert_bank(r, layer), x[i], b.ids[i],
                                             b.weights[i], work(r), s(r));
        } else {
            const std::size_t routed_bytes = Qwen4RoutedExperts::routed_workspace_bytes(T);
            if (routed_bytes != 0) { (void)work(r).alloc_bytes(routed_bytes); }
        }
        if (p.shared) {
            linear(r, x[i], p.shared->gate_up, b.gate_up[i]);
            if constexpr (kExecute) {
                ops::silu_mul(b.gate_up[i].slice(0, 0, shared_w),
                              b.gate_up[i].slice(0, shared_w, shared_w), b.activation[i], s(r));
            }
            linear(r, b.activation[i], p.shared->down, b.shared[i]);
        }
    });
}

template <class Arena>
void Qwen4Forward<Arena>::moe_combine(std::size_t layer, const RankTensors& x, MoeBuffers& b,
                                      RankTensors& y, const RankTensors& staging) {
    if constexpr (kExecute) {
        const std::int32_t T   = columns();
        const auto layer_index = static_cast<std::int32_t>(layer);
        // T62: at tp 2 a prompt chunk's MoE exchange may take the pipelined host exchange
        // (optionally a narrow payload, optionally by column groups that overlap the routed
        // experts of the later columns). Decode rounds (prefill == false) never do.
        const T62Exchange& t62 = t62_exchange();
        ops::HostExchange* pipe =
            width_ == 2 && c(0).prefill && t62.enabled ? tp_.events->host_exchange() : nullptr;
        if (pipe != nullptr &&
            !pipe->begin_sum_f32(b.partial, staging, &y, t62.payload, *tp_.execution)) {
            pipe = nullptr;
        }
        const OpenExchangeSum open_sum{pipe};
        each([&](int r) {
            const std::size_t i         = at(r);
            const Qwen4MoeParameters& p = block(r, layer).moe();
            const Qwen4Runtime& runtime = *state(r).runtime;
            if (runtime.moe_probe != nullptr && runtime.moe_probe->wants(layer_index)) {
                runtime.moe_probe->capture(layer_index, x[i], b.ids[i], b.weights[i], s(r));
            }
            // Columns [first, first + n) of this rank's partial are final: add the shared
            // expert there (owner rank; elementwise, the same bits as on the whole) and send.
            const auto ready = [&, r, i](std::int32_t first, std::int32_t n) {
                if (p.shared) {
                    Tensor part = b.partial[i].slice(1, first, n);
                    ops::moe_accumulate_shared(b.gate[i].slice(0, first, n),
                                               b.shared[i].slice(1, first, n), part, s(r));
                }
                pipe->columns_ready(r, first, n);
            };
            Qwen4ColumnGroups groups;
            const ColumnGroupsScope published(pipe != nullptr && t62.group_columns > 0 ? &groups
                                                                                       : nullptr);
            if (pipe != nullptr && t62.group_columns > 0) {
                groups.group_columns = t62.group_columns;
                groups.ready         = ready;
            }
            routed_experts(r, layer)->run(layer_index, expert_bank(r, layer), x[i], b.ids[i],
                                          b.weights[i], b.partial[i], work(r), s(r));
            if (width_ == 1) {
                ops::moe_combine_shared(b.partial[i], b.gate[i], b.shared[i], y[i], s(r));
            } else if (pipe != nullptr) {
                if (groups.done < T) { ready(groups.done, T - groups.done); }
            } else if (p.shared) {
                ops::moe_accumulate_shared(b.gate[i], b.shared[i], b.partial[i], s(r));
            }
        });
        if (width_ == 2) {
            if (pipe != nullptr) {
                pipe->finish_sum_f32();
            } else {
                if (Qwen4Taps* taps = exchange_taps()) {
                    taps->capture_exchange(layer_index, 1, b.partial[0], b.partial[1],
                                           c(0).first_position, s(0), s(1));
                }
                ops::allreduce_sum_f32(b.partial, staging, &y, *tp_.execution, *tp_.events);
            }
            each([&](int r) { routed_experts(r, layer)->exchanged(layer_index, s(r)); });
        }
    } else {
        (void)layer, (void)x, (void)b, (void)y, (void)staging;
    }
}

template <class Arena>
auto Qwen4Forward<Arena>::t62_exchange() -> const T62Exchange& {
    static const T62Exchange value = [] {
        T62Exchange v;
        const auto get = [](const char* name) -> std::string {
            const char* e = std::getenv(name);
            return e != nullptr ? std::string(e) : std::string();
        };
        if (get("NINFER_TP_AR") == "0") { return v; } // one allreduce_sum_f32 call
        const std::string payload = get("NINFER_TP_AR_PAYLOAD");
        if (payload == "fp32") { v.payload = ops::HostExchange::Payload::FP32; }
        if (payload == "fp16") { v.payload = ops::HostExchange::Payload::FP16; }
        if (payload == "bf16") { v.payload = ops::HostExchange::Payload::BF16; }
        if (payload == "int8") { v.payload = ops::HostExchange::Payload::INT8; }
        const std::string group = get("NINFER_TP_AR_GROUP");
        if (!group.empty()) {
            const auto g    = static_cast<std::int32_t>(std::strtol(group.c_str(), nullptr, 10));
            const auto unit = ops::kMoeGroupedChunkColumns;
            v.group_columns = g > 0 ? std::max(unit, g / unit * unit) : 0;
        }
        v.enabled = v.payload != ops::HostExchange::Payload::FP32 || v.group_columns > 0 ||
                    get("NINFER_TP_AR_PIPE") == "1";
        return v;
    }();
    return value;
}

template <class Arena>
Qwen4RoutedExperts* Qwen4Forward<Arena>::routed_experts(int r, std::size_t layer) const {
    const Qwen4Runtime* runtime = state(r).runtime;
    Qwen4RoutedExperts* experts =
        runtime == nullptr ? nullptr
                           : (layer == mtp_layer() ? runtime->mtp_experts : runtime->experts);
    if (experts == nullptr) {
        throw std::logic_error(layer == mtp_layer()
                                   ? "Qwen4Exp MTP layer has no bound expert backend"
                                   : "Qwen4Exp MoE has no routed-expert backend");
    }
    return experts;
}

template <class Arena>
const Qwen4ExpertBank& Qwen4Forward<Arena>::expert_bank(int r, std::size_t layer) const {
    if (layer != mtp_layer()) { return block(r, layer).moe().experts; }
    const Qwen4Runtime* runtime = state(r).runtime;
    if (runtime == nullptr || runtime->mtp_expert_bank == nullptr) {
        throw std::logic_error("Qwen4Exp MTP layer has no bound expert bank");
    }
    return *runtime->mtp_expert_bank;
}

} // namespace ninfer::models::qwen3_5::execution
