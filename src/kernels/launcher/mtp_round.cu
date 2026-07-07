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

void mtp_accept_tokens_launch(const Tensor& target_tokens, const Tensor& logits,
                              const Tensor& drafts, Tensor& length, Tensor& token,
                              Tensor& sampled_out, Tensor& num_sampled, Tensor& accepted,
                              Tensor& ar_pos, Tensor& stats, const SamplingConfig* config,
                              cudaStream_t stream) {
    const std::int32_t vocab = logits.ne[0];
    const std::int32_t cols = drafts.ne[0] + 1;
    const std::int32_t partial_blocks =
        (vocab + kSamplerPartialTileItems - 1) / kSamplerPartialTileItems;
    const std::int32_t groups = sampler_group_count(partial_blocks);
    // Same shared predicate the accept kernel self-guards on, so exactly one of
    // the single-block and multi-block paths runs for any shape.
    if (!sampler_multiblock_ok(vocab, cols, partial_blocks, groups)) {
        mtp_accept_tokens_kernel<<<1, kSamplerBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(target_tokens.data),
            static_cast<const __nv_bfloat16*>(logits.data),
            static_cast<const std::int32_t*>(drafts.data), static_cast<std::int32_t*>(length.data),
            static_cast<std::int32_t*>(token.data), static_cast<std::int32_t*>(sampled_out.data),
            static_cast<std::int32_t*>(num_sampled.data), static_cast<std::int32_t*>(accepted.data),
            static_cast<std::int32_t*>(ar_pos.data), static_cast<std::int64_t*>(stats.data),
            config, vocab, drafts.ne[0]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(cols));
    mtp_sampling_partial_topk_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), config, vocab);
    CUDA_CHECK(cudaGetLastError());
    const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(cols));
    mtp_sampling_group_finalize_kernel<<<group_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(target_tokens.data),
        static_cast<const std::int32_t*>(drafts.data), static_cast<std::int32_t*>(length.data),
        static_cast<std::int32_t*>(token.data), static_cast<std::int32_t*>(sampled_out.data),
        static_cast<std::int32_t*>(num_sampled.data), static_cast<std::int32_t*>(accepted.data),
        static_cast<std::int32_t*>(ar_pos.data), static_cast<std::int64_t*>(stats.data), config,
        vocab, cols, partial_blocks, groups);
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

void mtp_remap_draft_token_launch(Tensor& draft_token, const std::int32_t* id_map, std::int32_t n,
                                  cudaStream_t stream) {
    mtp_remap_draft_token_kernel<<<1, 1, 0, stream>>>(
        static_cast<std::int32_t*>(draft_token.data), id_map, n);
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
