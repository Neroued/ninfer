#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void launch_fp8_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_fp8_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_fp8_vocabulary_a16_mma(const Tensor& x, const Weight& weight, Tensor& out,
                                   cudaStream_t stream);

#ifdef NINFER_VOLTA_BUILD
// Quadpair-split-N form (fp8_volta_qpn_gemm.cuh), for the narrow verify widths where the SIMT
// A16 family decays and the groupwise routes do not. The tile height is mirrored here because
// dispatch is host code and cannot see the device header; the launcher static_asserts the two
// agree.
inline constexpr std::int32_t kFp8VoltaQpnRowsPerTile = 8;
inline constexpr std::int32_t kFp8VoltaQpnMaxTokens   = 32;
void launch_fp8_volta_qpn(const Tensor&, const Weight&, Tensor&, cudaStream_t);
[[nodiscard]] bool fp8_volta_qpn_supported(std::int32_t n, std::int32_t k,
                                           std::int32_t t) noexcept;
#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
