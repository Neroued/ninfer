// Implements: include/ninfer/ops/speculative_round.h
// Match: validated speculative state and BF16 verification logits.
// Algorithm assumptions: the shared sampling layout selects either one block
// or a two-launch partial/group pipeline without host reads of device config.
#include "ops/launcher/speculative_round.h"

#include "ops/common/math.h"
#include "ops/kernel/speculative_round.cuh"
#include "core/device.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

void speculative_prepare_verify_inputs_launch(const Tensor& anchors, const Tensor& drafts,
                                              const Tensor& base_positions,
                                              const Tensor& current_extents, Tensor& verify_ids,
                                              Tensor& positions, cudaStream_t stream) {
    constexpr int kBlock = 32;
    const int k          = drafts.ne[0];
    const int batch      = drafts.ne[1];
    const dim3 grid(static_cast<unsigned int>(div_up(k + 1, kBlock)),
                    static_cast<unsigned int>(batch));
    speculative_prepare_verify_inputs_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(base_positions.data),
        static_cast<const std::int32_t*>(current_extents.data),
        static_cast<std::int32_t*>(verify_ids.data), static_cast<std::int32_t*>(positions.data), k);
    CUDA_CHECK(cudaGetLastError());
}

void speculative_prepare_verify_ids_launch(const Tensor& anchors, const Tensor& drafts,
                                           const Tensor& current_extents, Tensor& verify_ids,
                                           cudaStream_t stream) {
    constexpr int kBlock = 32;
    const int k          = drafts.ne[0];
    const int batch      = drafts.ne[1];
    const dim3 grid(static_cast<unsigned int>(div_up(k + 1, kBlock)),
                    static_cast<unsigned int>(batch));
    speculative_prepare_verify_inputs_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const std::int32_t*>(drafts.data), nullptr,
        static_cast<const std::int32_t*>(current_extents.data),
        static_cast<std::int32_t*>(verify_ids.data), nullptr, k);
    CUDA_CHECK(cudaGetLastError());
}

void speculative_accept_greedy_drafts_launch(const Tensor& target_tokens, const Tensor& logits,
                                             const Tensor& drafts, const Tensor& current_extents,
                                             Tensor& lengths, Tensor& anchors,
                                             Tensor& licensed_tokens, Tensor& licensed_counts,
                                             Tensor& accepted, std::int32_t token_domain,
                                             const SamplingConfig* configs, DeviceSpan workspace,
                                             cudaStream_t stream) {
    const std::int32_t physical_rows     = logits.ne[0];
    const std::int32_t cols              = drafts.ne[0] + 1;
    const std::int32_t batch             = drafts.ne[1];
    const SamplingWorkspaceLayout layout = make_sampling_workspace_layout(token_domain, cols);
    if (!layout.multiblock) {
        speculative_accept_greedy_drafts_kernel<<<batch, kSamplerBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(target_tokens.data),
            static_cast<const __nv_bfloat16*>(logits.data),
            static_cast<const std::int32_t*>(drafts.data),
            static_cast<const std::int32_t*>(current_extents.data),
            static_cast<std::int32_t*>(lengths.data), static_cast<std::int32_t*>(anchors.data),
            static_cast<std::int32_t*>(licensed_tokens.data),
            static_cast<std::int32_t*>(licensed_counts.data),
            static_cast<std::int32_t*>(accepted.data), configs, token_domain, physical_rows,
            drafts.ne[0]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const std::int32_t partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const std::int32_t groups         = sampler_group_count(partial_blocks);
    const SamplingWorkspace scratch   = layout.bind(workspace);
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(cols), static_cast<unsigned int>(batch));
    speculative_sampling_partial_topk_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(current_extents.data), configs, token_domain,
        physical_rows, cols, drafts.ne[0], scratch, layout.bytes);
    CUDA_CHECK(cudaGetLastError());
    const dim3 batched_group_grid(static_cast<unsigned int>(groups),
                                  static_cast<unsigned int>(cols),
                                  static_cast<unsigned int>(batch));
    speculative_sampling_group_finalize_kernel<false>
        <<<batched_group_grid, kSamplerGroupBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(target_tokens.data),
            static_cast<const std::int32_t*>(drafts.data), nullptr, nullptr,
            static_cast<const std::int32_t*>(current_extents.data),
            static_cast<std::int32_t*>(lengths.data), static_cast<std::int32_t*>(anchors.data),
            static_cast<std::int32_t*>(licensed_tokens.data),
            static_cast<std::int32_t*>(licensed_counts.data),
            static_cast<std::int32_t*>(accepted.data), configs, token_domain, cols, partial_blocks,
            groups, scratch, layout.bytes);
    CUDA_CHECK(cudaGetLastError());
}

void speculative_accept_sparse_drafts_launch(
    const Tensor& target_tokens, const Tensor& logits, const Tensor& drafts,
    const Tensor& candidate_ids, const Tensor& proposal_q, const Tensor& current_extents,
    Tensor& round_lengths, Tensor& round_anchors, Tensor& licensed_tokens, Tensor& licensed_counts,
    Tensor& accepted_drafts, std::int32_t token_domain, const SamplingConfig* configs,
    bool raw_greedy, DeviceSpan workspace, cudaStream_t stream) {
    const std::int32_t batch = drafts.ne[1];
    const std::int32_t k     = drafts.ne[0];
    const std::int32_t cols  = k + 1;
    if (raw_greedy) {
        speculative_accept_sparse_warp_greedy_kernel<<<1, 32 * batch, 0, stream>>>(
            static_cast<const int*>(target_tokens.data), static_cast<const int*>(drafts.data),
            static_cast<const int*>(current_extents.data), static_cast<int*>(round_lengths.data),
            static_cast<int*>(round_anchors.data), static_cast<int*>(licensed_tokens.data),
            static_cast<int*>(licensed_counts.data), static_cast<int*>(accepted_drafts.data), k);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const SamplingWorkspaceLayout layout = make_sampling_workspace_layout(token_domain, cols);
    const std::int32_t partial_blocks    = div_up(token_domain, kSamplerPartialTileItems);
    const std::int32_t groups            = sampler_group_count(partial_blocks);
    const SamplingWorkspace scratch      = layout.bind(workspace);
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(cols), static_cast<unsigned int>(batch));
    speculative_sampling_partial_topk_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(current_extents.data), configs, token_domain, logits.ne[0],
        cols, k, scratch, layout.bytes);
    CUDA_CHECK(cudaGetLastError());

    const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(cols),
                          static_cast<unsigned int>(batch));

    speculative_sampling_group_finalize_kernel<true><<<group_grid, kSamplerGroupBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(target_tokens.data),
        static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(candidate_ids.data),
        static_cast<const float*>(proposal_q.data),
        static_cast<const std::int32_t*>(current_extents.data),
        static_cast<std::int32_t*>(round_lengths.data),
        static_cast<std::int32_t*>(round_anchors.data),
        static_cast<std::int32_t*>(licensed_tokens.data),
        static_cast<std::int32_t*>(licensed_counts.data),
        static_cast<std::int32_t*>(accepted_drafts.data), configs, token_domain, cols,
        partial_blocks, groups, scratch, layout.bytes);

    CUDA_CHECK(cudaGetLastError());
}

void speculative_select_accepted_hidden_launch(const Tensor& hidden, const Tensor& selectors,
                                               Tensor& out, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int rows       = hidden.ne[0];
    const int batch      = hidden.ne[2];
    const dim3 grid(static_cast<unsigned int>(std::max(1, div_up(rows, kBlock))),
                    static_cast<unsigned int>(batch));
    speculative_select_accepted_hidden_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data),
        static_cast<const std::int32_t*>(selectors.data), static_cast<__nv_bfloat16*>(out.data),
        rows, hidden.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void proposal_remap_token_ids_launch(Tensor& proposal_tokens, const std::int32_t* id_map,
                                     std::int32_t n, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int count      = proposal_tokens.ne[0];
    const int grid       = std::max(1, div_up(count, kBlock));
    proposal_remap_token_ids_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<std::int32_t*>(proposal_tokens.data), count, id_map, n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
