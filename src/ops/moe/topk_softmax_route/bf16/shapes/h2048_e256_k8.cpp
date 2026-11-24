#include "ops/moe/topk_softmax_route/dispatch.h"

#include <array>

namespace ninfer::ops::detail::moe_route {
namespace {
constexpr int kMmaFirst = 33;
constexpr std::array kBoundaries{2, kMmaFirst};
} // namespace

RouterPlan select_h2048_e256_k8(int tokens) {
    return {tokens == 1          ? RouterAlgorithm::Vector
            : tokens < kMmaFirst ? RouterAlgorithm::TokenTile
                                 : RouterAlgorithm::Mma,
            tokens > 1 && tokens < kMmaFirst, false};
}

std::span<const int> allocation_boundaries() { return kBoundaries; }
} // namespace ninfer::ops::detail::moe_route
