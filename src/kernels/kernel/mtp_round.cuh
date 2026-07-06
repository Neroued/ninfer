#pragma once

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

__global__ void mtp_accept_tokens_kernel(const std::int32_t* target_tokens,
                                         const std::int32_t* drafts, std::int32_t* length,
                                         std::int32_t* token, std::int32_t* sampled_out,
                                         std::int32_t* num_sampled, std::int32_t* accepted,
                                         std::int32_t* ar_pos, std::int64_t* stats,
                                         std::int32_t k) {
    int a = 0;
    while (a < k && target_tokens[a] == drafts[a]) { ++a; }
    const int t_star = target_tokens[a];

    for (int i = 0; i <= k; ++i) { sampled_out[i] = 0; }
    for (int i = 0; i < a; ++i) { sampled_out[i] = drafts[i]; }
    sampled_out[a] = t_star;

    const int produced = a + 1;
    *num_sampled      = produced;
    *accepted         = a;
    *token            = t_star;
    *length += produced;
    *ar_pos = *length;

    stats[0] += k;
    stats[1] += a;
    stats[2] += 1;
    for (int i = 0; i < a && i < kMtpRoundAcceptedPerPosLimit; ++i) {
        stats[kMtpRoundAcceptedPerPosOffset + i] += 1;
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
