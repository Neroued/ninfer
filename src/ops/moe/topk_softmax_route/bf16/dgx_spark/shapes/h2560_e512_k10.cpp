#include "ops/moe/topk_softmax_route/dispatch.h"

namespace ninfer::ops::detail::moe_route {
RouterPlan select_h2560_e512_k10(int tokens) {
    return {.algorithm    = tokens == 1 ? RouterAlgorithm::Vector : RouterAlgorithm::Mma,
            .mma_rows     = tokens >= 129 ? 64 : 16,
            .network_topk = tokens >= 129};
}
} // namespace ninfer::ops::detail::moe_route
