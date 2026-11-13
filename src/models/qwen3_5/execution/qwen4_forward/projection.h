#pragma once

// The closing projection of a block - into the hyper-connection block output or added into the
// plain residual - and its tp 2 row-parallel exchange: the BF16 all-reduce, the FP32 partials of
// the mixer output and the teacher-forcing taps of the exchanged partials.
//
// Member definitions of Qwen4Forward (qwen4_forward.h), included only by qwen4_forward.cpp, which
// instantiates them for execution and for planning.

#include "models/qwen3_5/execution/qwen4_forward.h"
#include "models/qwen3_5/execution/tp2/linear_split.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/tp2/linear_add.h"

#include <cstdlib>
#include <string>

namespace ninfer::models::qwen3_5::execution {

// Abandons the HostExchange sum a forward opened (null: none) when the work before its
// finish_sum_f32 throws, so a failed prompt chunk leaves the exchange free for the next one.
// finish_sum_f32 closes the call, and abandoning a closed call does nothing.
class OpenExchangeSum {
public:
    explicit OpenExchangeSum(ops::HostExchange* pipe) noexcept : pipe_(pipe) {}

    ~OpenExchangeSum() {
        if (pipe_ != nullptr) { pipe_->abandon_sum_f32(); }
    }

    OpenExchangeSum(const OpenExchangeSum&)            = delete;
    OpenExchangeSum& operator=(const OpenExchangeSum&) = delete;

private:
    ops::HostExchange* pipe_;
};

template <class Arena>
Qwen4Taps* Qwen4Forward<Arena>::exchange_taps() const {
    if constexpr (kExecute) {
        const Qwen4Runtime* runtime = state(0).runtime;
        if (width_ == 2 && c(0).prefill && runtime != nullptr && runtime->taps != nullptr &&
            runtime->taps->wants_exchange()) {
            return runtime->taps;
        }
    }
    return nullptr;
}

template <class Arena>
bool Qwen4Forward<Arena>::mix_fp32_partials() {
    static const bool on = [] {
        const char* v = std::getenv("NINFER_TP_MIX");
        return v == nullptr || std::string(v) != "bf16";
    }();
    return on;
}

template <class Arena>
void Qwen4Forward<Arena>::output_projection(const RankTensors& x,
                                            std::array<const LinearParameters*, 2> p,
                                            RankTensors& y, const RankTensors& staging,
                                            std::int32_t layer, int site) {
    if (width_ == 1) {
        linear(0, x[0], *p[0], y[0]);
        return;
    }
    if (mix_fp32_partials() && (p[0]->weight.qtype == QType::Q8_G32_FP16 ||
                                p[0]->weight.qtype == QType::MXFP8_E4M3FN_G32_UE8M0)) {
        const std::int32_t T = columns();
        auto scope           = scopes();
        RankTensors part     = alloc(DType::FP32, {hidden(), T});
        RankTensors stage    = alloc(DType::FP32, {hidden(), T});
        each([&](int r) { linear(r, x[at(r)], *p[at(r)], part[at(r)]); });
        if constexpr (kExecute) {
            (void)staging;
            if (Qwen4Taps* taps = exchange_taps()) { // T128c check: FP32 partials of the mixer
                taps->capture_exchange(layer, site, part[0], part[1], c(0).first_position, s(0),
                                       s(1));
            }
            ops::HostExchange* pipe = c(0).prefill ? tp_.events->host_exchange() : nullptr;
            if (pipe != nullptr &&
                pipe->begin_sum_f32(
                    part, stage, &y,
                    t62_exchange().payload == ops::HostExchange::Payload::INT8
                        ? ops::HostExchange::Payload::INT8 // same payload as the MoE exchange
                        : ops::HostExchange::Payload::FP16,
                    *tp_.execution)) {
                const OpenExchangeSum open_sum{pipe};
                pipe->columns_ready(0, 0, T);
                pipe->columns_ready(1, 0, T);
                pipe->finish_sum_f32();
            } else {
                ops::allreduce_sum_f32(part, stage, &y, *tp_.execution, *tp_.events);
            }
        }
        return;
    }
    if constexpr (kExecute) {
        if (Qwen4Taps* taps = exchange_taps()) {
            // linear_row_parallel unrolled: each rank's partial, the tap, the BF16 all-reduce.
            each([&](int r) { linear(r, x[at(r)], *p[at(r)], y[at(r)]); });
            taps->capture_exchange(layer, site, y[0], y[1], c(0).first_position, s(0), s(1));
            ops::allreduce_sum(y, staging, *tp_.execution, *tp_.events);
            return;
        }
        const std::array<WorkspaceArena*, 2> ws{&work(0), &work(1)};
        project_row_parallel(x, p, y, staging, ws, *tp_.execution, *tp_.events);
    } else {
        for (int r = 0; r < width_; ++r) {
            const LinearParameters& rp = *p[at(r)];
            reserve(ops::linear_workspace_capacity_bytes(rp.weight.qtype, rp.weight.n, rp.weight.k,
                                                         rp.policy, first_, last_),
                    r);
        }
    }
}

template <class Arena>
void Qwen4Forward<Arena>::close(const RankTensors& in, std::array<const LinearParameters*, 2> p,
                                RankTensors& out, const RankTensors& staging, std::int32_t layer,
                                int site) {
    if (has_hc()) {
        output_projection(in, p, out, staging, layer, site);
        return;
    }
    if constexpr (kExecute) {
        if (width_ == 1) {
            ops::linear_add(in[0], p[0]->weight, out[0], p[0]->policy, work(0), s(0));
            return;
        }
        const std::array<WorkspaceArena*, 2> ws{&work(0), &work(1)};
        if (slices_.usable()) {
            // Slice s's projection begins its all-reduce on lane s (plain_ffn finishes it).
            const std::int32_t T = columns();
            for (std::int32_t slice = 0; slice < slices_.count; ++slice) {
                const std::int32_t first = slices_.first(slice);
                const std::int32_t count = slices_.columns(slice, T);
                RankTensors residual     = window(out, first, count);
                project_add_row_parallel(window(in, first, count), p, residual,
                                         window(staging, first, count), ws, *tp_.execution,
                                         *tp_.events, static_cast<int>(slice));
            }
            return;
        }
        // Under the half-chunk overlap the all-reduce is only begun, on the half's lane.
        project_add_row_parallel(in, p, out, staging, ws, *tp_.execution, *tp_.events, lane_);
    } else {
        (void)staging, (void)layer, (void)site, (void)out, (void)in;
        for (int r = 0; r < width_; ++r) {
            const Weight& w = p[at(r)]->weight;
            reserve(width_ == 1 ? ops::linear_add_workspace_capacity_bytes(
                                      w.qtype, w.n, w.k, p[at(r)]->policy, first_, last_)
                                : ops::linear_add_row_parallel_workspace_capacity_bytes(
                                      w.qtype, w.n, w.k, p[at(r)]->policy, first_, last_),
                    r);
        }
    }
}

} // namespace ninfer::models::qwen3_5::execution
