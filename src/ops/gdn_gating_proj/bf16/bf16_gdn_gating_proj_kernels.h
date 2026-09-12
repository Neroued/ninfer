#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

void bf16_gdn_norm_gating_proj_27_launch(const Tensor& x, const Tensor& norm_weight, float eps,
                                         Tensor& h, const Weight& a_weight, const Weight& b_weight,
                                         const Tensor& alog, const Tensor& bias, Tensor& g,
                                         Tensor& beta, cudaStream_t stream);

enum class Bf16GdnGatingTokenVariant {
    None,
    Full,
    Predicated,
};

void bf16_gdn_gating_proj_gemv_launch(const Tensor& x, const Weight& a_weight,
                                      const Weight& b_weight, const Tensor& A_log,
                                      const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                      cudaStream_t stream);
void bf16_gdn_gating_proj_small_t_split10_launch(const Tensor& x, const Weight& a_weight,
                                                 const Weight& b_weight, const Tensor& A_log,
                                                 const Tensor& dt_bias, void* workspace,
                                                 std::size_t workspace_bytes, Tensor& g,
                                                 Tensor& beta, cudaStream_t stream);

// Cooperative launchers return false without submitting work only when the selected device cannot
// make one complete token tile resident. The Op wrapper owns the non-cooperative fallback.
[[nodiscard]] bool bf16_gdn_gating_proj_mma_split8_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Weight& a_weight,
    const Weight& b_weight, const Tensor& A_log, const Tensor& dt_bias, void* workspace, Tensor& g,
    Tensor& beta, std::int32_t multiprocessor_count, cudaStream_t stream);
[[nodiscard]] bool bf16_gdn_gating_proj_mma_split4_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Weight& a_weight,
    const Weight& b_weight, const Tensor& A_log, const Tensor& dt_bias, void* workspace, Tensor& g,
    Tensor& beta, std::int32_t multiprocessor_count, cudaStream_t stream);
[[nodiscard]] bool bf16_gdn_gating_proj_mma_split2_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Weight& a_weight,
    const Weight& b_weight, const Tensor& A_log, const Tensor& dt_bias, void* workspace, Tensor& g,
    Tensor& beta, std::int32_t multiprocessor_count, cudaStream_t stream);
void bf16_gdn_gating_proj_mma_unsplit_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                             const Weight& a_weight, const Weight& b_weight,
                                             const Tensor& A_log, const Tensor& dt_bias, Tensor& g,
                                             Tensor& beta, cudaStream_t stream);

void bf16_gdn_gating_proj_35_simt_c4_launch(const Tensor& x, const Weight& a_weight,
                                            const Weight& b_weight, const Tensor& A_log,
                                            const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                            cudaStream_t stream);
void bf16_gdn_gating_proj_35_simt_c8_launch(const Tensor& x, const Weight& a_weight,
                                            const Weight& b_weight, const Tensor& A_log,
                                            const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                            cudaStream_t stream);
[[nodiscard]] bool bf16_gdn_gating_proj_35_mma_split32_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Weight& a_weight,
    const Weight& b_weight, const Tensor& A_log, const Tensor& dt_bias, void* workspace, Tensor& g,
    Tensor& beta, std::int32_t multiprocessor_count, cudaStream_t stream);
[[nodiscard]] bool bf16_gdn_norm_gating_proj_35_mma_split32_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Tensor& norm_weight, float eps,
    Tensor& h, const Weight& a_weight, const Weight& b_weight, const Tensor& A_log,
    const Tensor& dt_bias, void* workspace, Tensor& g, Tensor& beta,
    std::int32_t multiprocessor_count, cudaStream_t stream);
[[nodiscard]] bool bf16_gdn_gating_proj_35_mma_split16_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Weight& a_weight,
    const Weight& b_weight, const Tensor& A_log, const Tensor& dt_bias, void* workspace, Tensor& g,
    Tensor& beta, std::int32_t multiprocessor_count, cudaStream_t stream);
[[nodiscard]] bool bf16_gdn_gating_proj_35_mma_split8_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Weight& a_weight,
    const Weight& b_weight, const Tensor& A_log, const Tensor& dt_bias, void* workspace, Tensor& g,
    Tensor& beta, std::int32_t multiprocessor_count, cudaStream_t stream);
[[nodiscard]] bool bf16_gdn_gating_proj_35_mma_split4_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Weight& a_weight,
    const Weight& b_weight, const Tensor& A_log, const Tensor& dt_bias, void* workspace, Tensor& g,
    Tensor& beta, std::int32_t multiprocessor_count, cudaStream_t stream);
[[nodiscard]] bool bf16_gdn_gating_proj_35_mma_split2_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Weight& a_weight,
    const Weight& b_weight, const Tensor& A_log, const Tensor& dt_bias, void* workspace, Tensor& g,
    Tensor& beta, std::int32_t multiprocessor_count, cudaStream_t stream);
void bf16_gdn_gating_proj_35_mma_unsplit_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                                const Weight& a_weight, const Weight& b_weight,
                                                const Tensor& A_log, const Tensor& dt_bias,
                                                Tensor& g, Tensor& beta, cudaStream_t stream);

} // namespace ninfer::ops::detail
