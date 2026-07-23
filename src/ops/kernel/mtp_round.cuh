#pragma once

// Implements: include/ninfer/ops/mtp_round.h
// Match: the fixed K=1..5 autoregressive MTP alignment window.

#include <cstdint>

namespace ninfer::ops {

__global__ void mtp_prepare_alignment_ids_kernel(const std::int32_t* verify_ids,
                                                 const std::int32_t* token,
                                                 const std::int32_t* accepted,
                                                 std::int32_t* alignment_ids, std::int32_t T) {
    const int k = T - 1;
    for (int i = 0; i < k; ++i) { alignment_ids[i] = verify_ids[i + 1]; }
    alignment_ids[*accepted] = *token;
}

} // namespace ninfer::ops
