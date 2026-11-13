#pragma once

// The forward's construction and the operands every block uses: the bound Parameters'
// FFN kinds, the residual buffers, the per-rank Linear, the row copies and the teacher-forcing
// taps.
//
// Member definitions of Qwen4Forward (qwen4_forward.h), included only by qwen4_forward.cpp, which
// instantiates them for execution and for planning.

#include "models/qwen3_5/execution/qwen4_forward.h"
#include "core/device.h"
#include "core/device_scope.h"
#include "ninfer/ops/copy_rows.h"
#include "ninfer/ops/linear.h"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

namespace ninfer::models::qwen3_5::execution {

template <class Arena>
Qwen4Forward<Arena>::Qwen4Forward(const TextConfig& config, std::array<Rank, 2> ranks, int width,
                                  Qwen4Collectives collectives, std::int32_t first,
                                  std::int32_t last)
    : config_(config), r_(ranks), width_(width), tp_(collectives), first_(first), last_(last) {
    if (first_ <= 0 || last_ < first_) {
        throw std::invalid_argument("Text forward column interval is invalid");
    }
    if (width_ != 1 && width_ != 2) {
        throw std::invalid_argument("Text forward width must be 1 or 2");
    }
    if (r_[0].columns.replay_records != nullptr && r_[0].columns.prefill) {
        throw std::invalid_argument("Text forward speculative records require a decode window");
    }
    for (int r = 0; r < width_; ++r) {
        const Rank& rank = r_[static_cast<std::size_t>(r)];
        if (rank.config == nullptr || rank.parameters == nullptr || rank.work == nullptr ||
            !rank.parameters->qwen4 ||
            rank.parameters->qwen4->layers.size() != config_.layer_types.size()) {
            throw std::invalid_argument("Text forward rank " + std::to_string(r) +
                                        " has no Text forward Parameters");
        }
    }
    if constexpr (kExecute) {
        if (width_ == 2 &&
            (tp_.execution == nullptr || tp_.events == nullptr || tp_.execution->tp != 2)) {
            throw std::invalid_argument("Text forward at tp 2 has no collectives");
        }
    }
    for (std::size_t layer = 0; layer < config_.layer_types.size(); ++layer) {
        require_ffn(block_ffns(layer), "text/layers/" + std::to_string(layer));
    }
}

template <class Arena>
std::array<const Qwen4FfnParameters*, 2> Qwen4Forward<Arena>::block_ffns(std::size_t layer) const {
    return {&qwen4(0).layers[layer].ffn, width_ == 2 ? &qwen4(1).layers[layer].ffn : nullptr};
}

template <class Arena>
void Qwen4Forward<Arena>::require_ffn(std::array<const Qwen4FfnParameters*, 2> ffn,
                                      const std::string& name) const {
    int holders = 0;
    for (int r = 0; r < width_; ++r) {
        const auto* moe = std::get_if<Qwen4MoeParameters>(ffn[at(r)]);
        if ((moe != nullptr) != moe_ffn()) {
            throw std::invalid_argument(name + ": the FFN parameters differ from the config's "
                                               "FFN kind");
        }
        holders += moe != nullptr && moe->shared ? 1 : 0;
    }
    if (moe_ffn() && holders != 1) {
        throw std::invalid_argument(name + ": the shared expert must be held by exactly one rank");
    }
}

template <class Arena>
auto Qwen4Forward<Arena>::wide_buffers(RankTensors& wide_out, std::int32_t T) -> RankTensors {
    RankTensors out;
    if constexpr (kExecute) {
        for (int r = 0; r < width_; ++r) {
            Tensor& w = wide_out[at(r)];
            if (w.dtype != DType::BF16 || w.ne[0] != wide_rows() ||
                w.numel() != static_cast<std::int64_t>(wide_rows()) * T || !w.is_contiguous() ||
                w.data == nullptr) {
                throw std::invalid_argument(
                    "Qwen4Exp wide residual output must be BF16 [S*H,T] on every rank");
            }
            out[at(r)] = w.view({wide_rows(), T});
        }
    } else {
        (void)wide_out;
        (void)T;
    }
    return out;
}

template <class Arena>
Qwen4Taps* Qwen4Forward<Arena>::prompt_taps(std::int32_t layer) const {
    if constexpr (kExecute) {
        const Qwen4Runtime* runtime = state(0).runtime;
        if (c(0).prefill && runtime != nullptr && runtime->taps != nullptr &&
            runtime->taps->wants(layer)) {
            return runtime->taps;
        }
    } else {
        (void)layer;
    }
    return nullptr;
}

template <class Arena>
void Qwen4Forward<Arena>::capture_taps(Qwen4Taps& taps, std::int32_t layer, const RankTensors& x) {
    std::optional<ScopedCurrentDevice> rank0;
    if (width_ == 2) { rank0.emplace(r_[0].device); }
    taps.capture(layer, x[0], c(0).first_position, s(0));
    if (width_ == 2) { taps.compare_ranks(layer, x[0], x[1], s(0), s(1)); }
}

template <class Arena>
void Qwen4Forward<Arena>::linear(int r, const Tensor& x, const LinearParameters& p, Tensor& out) {
    if constexpr (kExecute) {
        ops::linear(x, p.weight, out, p.policy, work(r), s(r));
    } else {
        reserve(
            ops::linear_workspace_capacity_bytes(p.weight.qtype, p.weight.n, p.weight.k, p.policy,
                                                 first_, last_) +
                ops::linear_optional_workspace_bytes(p.weight.qtype, p.weight.n, p.weight.k, last_),
            r);
    }
}

template <class Arena>
void Qwen4Forward<Arena>::copy_rows(int r, const Tensor& source, std::int32_t row0,
                                    std::int32_t rows, const Tensor& destination) {
    if constexpr (kExecute) {
        const std::size_t element  = dtype_size(source.dtype);
        const std::int64_t columns = source.numel() / source.ne[0];
        // T139: a kernel, not a memcpy node: cudaGraphExecUpdate sometimes rejected the 2-D
        // memcpy.
        ops::copy_rows(
            destination.data,
            static_cast<const std::byte*>(source.data) + static_cast<std::size_t>(row0) * element,
            static_cast<std::size_t>(source.ne[0]) * element,
            static_cast<std::size_t>(rows) * element, static_cast<std::size_t>(columns), s(r));
    }
}

template <class Arena>
Tensor Qwen4Forward<Arena>::split_rows(int r, const Tensor& source, std::int32_t row0,
                                       std::int32_t rows, const Tensor& destination) {
    if constexpr (kExecute) {
        if (source.numel() / source.ne[0] == 1) {
            Tensor view = destination;
            view.data   = static_cast<std::byte*>(source.data) +
                        static_cast<std::size_t>(row0) * dtype_size(source.dtype);
            return view;
        }
    }
    copy_rows(r, source, row0, rows, destination);
    return destination;
}

} // namespace ninfer::models::qwen3_5::execution
