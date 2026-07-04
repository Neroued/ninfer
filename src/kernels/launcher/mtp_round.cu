#include "kernels/launcher/mtp_round.h"

#include "kernels/kernel/mtp_round.cuh"
#include "qus/core/device.h"

#include <algorithm>
#include <cstdint>

namespace qus::kernels::detail {

void mtp_prepare_verify_inputs_launch(const Tensor& token, const Tensor& drafts,
                                      const Tensor& length, Tensor& window_base,
                                      Tensor& verify_ids, Tensor& positions,
                                      cudaStream_t stream) {
    constexpr int kBlock = 32;
    const int T          = drafts.ne[0] + 1;
    mtp_prepare_verify_inputs_kernel<<<1, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(token.data),
        static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(length.data),
        static_cast<std::int32_t*>(window_base.data),
        static_cast<std::int32_t*>(verify_ids.data),
        static_cast<std::int32_t*>(positions.data), T);
    CUDA_CHECK(cudaGetLastError());
}

void mtp_accept_tokens_launch(const Tensor& target_tokens, const Tensor& drafts, Tensor& length,
                              Tensor& token, Tensor& sampled_out, Tensor& num_sampled,
                              Tensor& accepted, Tensor& ar_pos, Tensor& stats,
                              cudaStream_t stream) {
    mtp_accept_tokens_kernel<<<1, 1, 0, stream>>>(
        static_cast<const std::int32_t*>(target_tokens.data),
        static_cast<const std::int32_t*>(drafts.data), static_cast<std::int32_t*>(length.data),
        static_cast<std::int32_t*>(token.data), static_cast<std::int32_t*>(sampled_out.data),
        static_cast<std::int32_t*>(num_sampled.data), static_cast<std::int32_t*>(accepted.data),
        static_cast<std::int32_t*>(ar_pos.data), static_cast<std::int64_t*>(stats.data),
        drafts.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

void mtp_prepare_shifted_ids_launch(const Tensor& verify_ids, const Tensor& token,
                                    const Tensor& accepted, Tensor& shifted_ids,
                                    cudaStream_t stream) {
    mtp_prepare_shifted_ids_kernel<<<1, 1, 0, stream>>>(
        static_cast<const std::int32_t*>(verify_ids.data),
        static_cast<const std::int32_t*>(token.data),
        static_cast<const std::int32_t*>(accepted.data),
        static_cast<std::int32_t*>(shifted_ids.data), verify_ids.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

void mtp_gather_hidden_row_launch(const Tensor& hidden, const Tensor& accepted, Tensor& out,
                                  cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int rows       = hidden.ne[0];
    const int grid       = static_cast<int>(std::max(1, (rows + kBlock - 1) / kBlock));
    mtp_gather_hidden_row_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data),
        static_cast<const std::int32_t*>(accepted.data), static_cast<__nv_bfloat16*>(out.data),
        rows, hidden.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void mtp_increment_i32_launch(Tensor& scalar, cudaStream_t stream) {
    mtp_increment_i32_kernel<<<1, 1, 0, stream>>>(static_cast<std::int32_t*>(scalar.data));
    CUDA_CHECK(cudaGetLastError());
}

void mtp_count_fallback_step_launch(Tensor& stats, cudaStream_t stream) {
    mtp_count_fallback_step_kernel<<<1, 1, 0, stream>>>(
        static_cast<std::int64_t*>(stats.data));
    CUDA_CHECK(cudaGetLastError());
}

void mtp_reset_gdn_initial_slot_launch(Tensor& gdn_initial_slot, cudaStream_t stream) {
    mtp_reset_gdn_initial_slot_kernel<<<1, 1, 0, stream>>>(
        static_cast<std::int32_t*>(gdn_initial_slot.data));
    CUDA_CHECK(cudaGetLastError());
}

void mtp_set_gdn_initial_slot_from_accepted_launch(const Tensor& accepted, Tensor& gdn_initial_slot,
                                                   cudaStream_t stream) {
    mtp_set_gdn_initial_slot_from_accepted_kernel<<<1, 1, 0, stream>>>(
        static_cast<const std::int32_t*>(accepted.data),
        static_cast<std::int32_t*>(gdn_initial_slot.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
