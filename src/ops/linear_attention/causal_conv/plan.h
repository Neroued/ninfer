#pragma once

#include "core/tensor.h"
#include <cstdint>

namespace ninfer::ops::detail {
enum class CausalConvRoute { Decode, TokenParallel, PrefillScalar, PrefillPairs };
enum class CausalConvSnapshotRoute {
    Decode,
    TokenParallel,
    Sequence,
    BatchedParallel,
    BatchedSequence
};

struct CausalConvSnapshotPlan {
    CausalConvSnapshotRoute route;
    bool masked;
};

enum class CausalConvSplitGeometry { Rows2048x2048x4096, Rows2048x2048x6144, Rows1024x1024x3072 };

inline CausalConvRoute select_causal_conv_plan(const Tensor& x, const Tensor& weight,
                                               const Tensor& state_in, const Tensor& state_out,
                                               const Tensor& out) {
    if (x.ne[1] == 1) return CausalConvRoute::Decode;
    if (x.ne[1] <= 32) return CausalConvRoute::TokenParallel;
    const auto addresses = reinterpret_cast<std::uintptr_t>(x.data) |
                           reinterpret_cast<std::uintptr_t>(weight.data) |
                           reinterpret_cast<std::uintptr_t>(state_in.data) |
                           reinterpret_cast<std::uintptr_t>(state_out.data) |
                           reinterpret_cast<std::uintptr_t>(out.data);
    return (addresses & 3) == 0 && (x.ne[0] & 1) == 0 ? CausalConvRoute::PrefillPairs
                                                      : CausalConvRoute::PrefillScalar;
}

inline CausalConvRoute select_causal_conv_split_plan(int tokens) {
    return tokens <= 32 ? CausalConvRoute::TokenParallel : CausalConvRoute::PrefillPairs;
}

inline CausalConvSnapshotPlan select_causal_conv_snapshot_plan(int tokens, int batch, bool masked) {
    if (batch == 1 && !masked) {
        if (tokens == 1) return {CausalConvSnapshotRoute::Decode, false};
        if (tokens <= 32) return {CausalConvSnapshotRoute::TokenParallel, false};
        return {CausalConvSnapshotRoute::Sequence, false};
    }
    return {tokens > 1 && tokens <= 32 ? CausalConvSnapshotRoute::BatchedParallel
                                       : CausalConvSnapshotRoute::BatchedSequence,
            masked};
}
} // namespace ninfer::ops::detail
