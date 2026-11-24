#include "ops/moe/topk_softmax_route/dispatch.h"

namespace ninfer::ops::detail::moe_route {
RouterPlan select_h2560_e512_k10(int tokens) {
    return {tokens == 1 ? RouterAlgorithm::Vector : RouterAlgorithm::Mma};
}
} // namespace ninfer::ops::detail::moe_route
