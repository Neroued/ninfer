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

void speculative_prepare_verify_inputs_launch(const Tensor& token, const Tensor& drafts,
                                              const Tensor& length, Tensor& verify_ids,
                                              Tensor& positions, cudaStream_t stream) {
    constexpr int kBlock = 32;
    const int T          = drafts.ne[0] + 1;
    speculative_prepare_verify_inputs_kernel<<<1, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(token.data), static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(length.data), static_cast<std::int32_t*>(verify_ids.data),
        static_cast<std::int32_t*>(positions.data), T);
    CUDA_CHECK(cudaGetLastError());
}

void speculative_accept_greedy_drafts_launch(const Tensor& target_tokens, const Tensor& logits,
                                             const Tensor& drafts, Tensor& length, Tensor& token,
                                             Tensor& sampled_out, Tensor& num_sampled,
                                             Tensor& accepted, Tensor& stats,
                                             std::int32_t token_domain,
                                             const SamplingConfig* config, DeviceSpan workspace,
                                             cudaStream_t stream) {
    const std::int32_t physical_rows     = logits.ne[0];
    const std::int32_t cols              = drafts.ne[0] + 1;
    const SamplingWorkspaceLayout layout = make_sampling_workspace_layout(token_domain, cols);
    if (!layout.multiblock) {
        speculative_accept_greedy_drafts_kernel<<<1, kSamplerBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(target_tokens.data),
            static_cast<const __nv_bfloat16*>(logits.data),
            static_cast<const std::int32_t*>(drafts.data), static_cast<std::int32_t*>(length.data),
            static_cast<std::int32_t*>(token.data), static_cast<std::int32_t*>(sampled_out.data),
            static_cast<std::int32_t*>(num_sampled.data), static_cast<std::int32_t*>(accepted.data),
            static_cast<std::int64_t*>(stats.data), config, token_domain, physical_rows,
            drafts.ne[0]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const std::int32_t partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const std::int32_t groups         = sampler_group_count(partial_blocks);
    const SamplingWorkspace scratch   = layout.bind(workspace);
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(cols));
    speculative_sampling_partial_topk_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<const std::int32_t*>(drafts.data), config, token_domain, physical_rows,
        scratch);
    CUDA_CHECK(cudaGetLastError());
    const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(cols));
    speculative_sampling_group_finalize_kernel<<<group_grid, kSamplerGroupBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(target_tokens.data),
        static_cast<const std::int32_t*>(drafts.data), static_cast<std::int32_t*>(length.data),
        static_cast<std::int32_t*>(token.data), static_cast<std::int32_t*>(sampled_out.data),
        static_cast<std::int32_t*>(num_sampled.data), static_cast<std::int32_t*>(accepted.data),
        static_cast<std::int64_t*>(stats.data), config, token_domain, cols, partial_blocks, groups,
        scratch);
    CUDA_CHECK(cudaGetLastError());
}

void speculative_select_accepted_hidden_launch(const Tensor& hidden, const Tensor& accepted,
                                               Tensor& out, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int rows       = hidden.ne[0];
    const int grid       = std::max(1, div_up(rows, kBlock));
    speculative_select_accepted_hidden_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data),
        static_cast<const std::int32_t*>(accepted.data), static_cast<__nv_bfloat16*>(out.data),
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
