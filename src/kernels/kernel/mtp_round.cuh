#pragma once

#include "kernels/kernel/sampling_device.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

inline constexpr int kMtpRoundAcceptedPerPosOffset = 4;
inline constexpr int kMtpRoundAcceptedPerPosLimit  = 5;

__global__ void mtp_prepare_verify_inputs_kernel(const std::int32_t* token,
                                                 const std::int32_t* drafts,
                                                 const std::int32_t* length,
                                                 std::int32_t* window_base,
                                                 std::int32_t* verify_ids,
                                                 std::int32_t* positions, std::int32_t T) {
    const int i = threadIdx.x;
    if (i == 0) { *window_base = *length; }
    if (i >= T) { return; }
    verify_ids[i] = i == 0 ? *token : drafts[i - 1];
    positions[i]  = *length + i;
}

// Commits the round's accepted tokens plus one correction/bonus token, then
// advances the length/ar_pos scalars and the acceptance stats. The greedy branch
// (config temperature <= 0) is bit-identical to the original argmax accept: keep
// the longest draft prefix whose target argmax matches, then take the target
// argmax at the divergence column. The sampling branch (temperature > 0) runs
// distribution-correct speculative rejection sampling over the verify logits with
// a one-hot (greedy) draft: accept drafts[i] with probability p_i(drafts[i]) under
// the truncated target distribution, resample from the masked residual on the
// first rejection, and draw a bonus from the last column when every draft accepts.
// The draft-proposal path stays greedy, so q is one-hot and the accept test
// collapses to `u < p_i(drafts[i])`. Launch with a single block of kSamplerBlock
// threads; only thread 0 performs the sequential accept/commit while the whole
// block cooperates on the per-column truncated-distribution build.
__launch_bounds__(kSamplerBlock) __global__ void mtp_accept_tokens_kernel(
    const std::int32_t* target_tokens, const __nv_bfloat16* logits, const std::int32_t* drafts,
    std::int32_t* length, std::int32_t* token, std::int32_t* sampled_out,
    std::int32_t* num_sampled, std::int32_t* accepted, std::int32_t* ar_pos, std::int64_t* stats,
    const SamplingConfig* cfg_ptr, std::int32_t vocab, std::int32_t k) {
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = *cfg_ptr;

    if (!(cfg.temperature > 0.0f)) {
        if (tid == 0) {
            int a = 0;
            while (a < k && target_tokens[a] == drafts[a]) { ++a; }
            const int t_star = target_tokens[a];

            for (int i = 0; i <= k; ++i) { sampled_out[i] = 0; }
            for (int i = 0; i < a; ++i) { sampled_out[i] = drafts[i]; }
            sampled_out[a] = t_star;

            const int produced = a + 1;
            *num_sampled       = produced;
            *accepted          = a;
            *token             = t_star;
            *length += produced;
            *ar_pos = *length;

            stats[0] += k;
            stats[1] += a;
            stats[2] += 1;
            for (int i = 0; i < a && i < kMtpRoundAcceptedPerPosLimit; ++i) {
                stats[kMtpRoundAcceptedPerPosOffset + i] += 1;
            }
        }
        return;
    }

    __shared__ float red_val[kSamplerBlock];
    __shared__ int red_idx[kSamplerBlock];
    __shared__ float cand_val[kSamplerMaxCandidates];
    __shared__ int cand_idx[kSamplerMaxCandidates];
    __shared__ float prob[kSamplerMaxCandidates];
    __shared__ int n_support;
    __shared__ int a_sh;
    __shared__ int done_sh;
    __shared__ int tstar_sh;
    __shared__ int L_sh;

    if (tid == 0) {
        a_sh    = 0;
        done_sh = 0;
        tstar_sh = 0;
        L_sh    = *length;
    }
    __syncthreads();

    for (int i = 0; i <= k; ++i) {
        sampling_build_truncated(logits, static_cast<std::int64_t>(i) * vocab, vocab, cfg, red_val,
                                 red_idx, cand_val, cand_idx, prob, &n_support);
        if (tid == 0 && done_sh == 0) {
            const int L = L_sh;
            if (i < k) {
                const int d = drafts[i];
                float pd    = 0.0f;
                for (int j = 0; j < n_support; ++j) {
                    if (cand_idx[j] == d) {
                        pd = prob[j];
                        break;
                    }
                }
                const float u =
                    sampling_uniform(cfg.seed, L + i + 1, kSamplePurposeMtpAccept, 0u);
                if (u < pd) {
                    a_sh = i + 1;  // accept drafts[i], keep verifying
                } else {
                    const float ur =
                        sampling_uniform(cfg.seed, L + i + 1, kSamplePurposeMtpResample, 0u);
                    tstar_sh = sampling_pick_from_support(cand_idx, prob, n_support, d, ur);
                    done_sh  = 1;
                }
            } else {
                // Every draft accepted: bonus token from the last verify column.
                const float u =
                    sampling_uniform(cfg.seed, L + k + 1, kSamplePurposeMtpBonus, 0u);
                tstar_sh = sampling_pick_from_support(cand_idx, prob, n_support, -1, u);
                done_sh  = 1;
            }
        }
        __syncthreads();
        if (done_sh) { break; }
    }

    if (tid == 0) {
        const int a     = a_sh;
        const int tstar = tstar_sh;
        const int L     = L_sh;

        for (int i = 0; i <= k; ++i) { sampled_out[i] = 0; }
        for (int i = 0; i < a; ++i) { sampled_out[i] = drafts[i]; }
        sampled_out[a] = tstar;

        const int produced = a + 1;
        *num_sampled       = produced;
        *accepted          = a;
        *token             = tstar;
        *length            = L + produced;
        *ar_pos            = *length;

        stats[0] += k;
        stats[1] += a;
        stats[2] += 1;
        for (int i = 0; i < a && i < kMtpRoundAcceptedPerPosLimit; ++i) {
            stats[kMtpRoundAcceptedPerPosOffset + i] += 1;
        }
        if (cfg.token_counts != nullptr) {
            for (int i = 0; i < produced; ++i) { atomicAdd(&cfg.token_counts[sampled_out[i]], 1); }
        }
    }
}

__global__ void mtp_prepare_shifted_ids_kernel(const std::int32_t* verify_ids,
                                               const std::int32_t* token,
                                               const std::int32_t* accepted,
                                               std::int32_t* shifted_ids, std::int32_t T) {
    const int k = T - 1;
    for (int i = 0; i < k; ++i) { shifted_ids[i] = verify_ids[i + 1]; }
    shifted_ids[*accepted] = *token;
}

__global__ void mtp_gather_hidden_row_kernel(const __nv_bfloat16* hidden,
                                             const std::int32_t* accepted, __nv_bfloat16* out,
                                             std::int32_t rows, std::int32_t cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) { return; }
    const int col = *accepted;
    if (col < 0 || col >= cols) { return; }
    out[row] = hidden[static_cast<std::int64_t>(col) * rows + row];
}

__global__ void mtp_remap_draft_token_kernel(std::int32_t* draft_token,
                                             const std::int32_t* id_map, std::int32_t n) {
    const int idx = *draft_token;
    if (idx >= 0 && idx < n) { *draft_token = id_map[idx]; }
}

__global__ void mtp_increment_i32_kernel(std::int32_t* scalar) { *scalar += 1; }

__global__ void mtp_count_fallback_step_kernel(std::int64_t* stats) { stats[3] += 1; }

__global__ void mtp_reset_gdn_initial_slot_kernel(std::int32_t* gdn_initial_slot) {
    *gdn_initial_slot = 0;
}

__global__ void mtp_set_gdn_initial_slot_from_accepted_kernel(const std::int32_t* accepted,
                                                              std::int32_t* gdn_initial_slot) {
    *gdn_initial_slot = *accepted;
}

} // namespace qus::kernels
