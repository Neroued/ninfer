// Implements: include/ninfer/ops/mtp_round.h
// Match: validated K=1..5 MTP alignment state.
#include "ops/launcher/mtp_round.h"

#include "core/device.h"
#include "ops/kernel/mtp_round.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

void mtp_prepare_alignment_ids_launch(const Tensor& verify_ids, const Tensor& token,
                                      const Tensor& accepted, Tensor& alignment_ids,
                                      cudaStream_t stream) {
    mtp_prepare_alignment_ids_kernel<<<1, 1, 0, stream>>>(
        static_cast<const std::int32_t*>(verify_ids.data),
        static_cast<const std::int32_t*>(token.data),
        static_cast<const std::int32_t*>(accepted.data),
        static_cast<std::int32_t*>(alignment_ids.data), verify_ids.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
