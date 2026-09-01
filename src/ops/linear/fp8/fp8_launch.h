#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void launch_fp8_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_fp8_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_fp8_vocabulary_a16_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream);
void launch_fp8_vocabulary_a16_gemm(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream);

#ifdef NINFER_VOLTA_BUILD
inline constexpr std::int32_t kFp8VoltaQpnRowsPerTile = 8;
inline constexpr std::int32_t kFp8VoltaQpnMaxTokens   = 32;
void launch_fp8_volta_qpn(const Tensor& x, const Weight& weight, Tensor& out,
                          cudaStream_t stream);
[[nodiscard]] bool fp8_volta_qpn_supported(std::int32_t n, std::int32_t k,
                                           std::int32_t t) noexcept;
#endif

} // namespace ninfer::ops::detail
