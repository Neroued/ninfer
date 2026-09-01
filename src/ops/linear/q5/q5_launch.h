#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

using Q5Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

#ifdef NINFER_VOLTA_BUILD
// Fused-dequant tensor-core route (q5_volta_mma_gemm.cuh). Not in the Q5Launch table: split-K
// needs an fp32 accumulator, and the linear_add form needs the residual flag.
// `splits_override` is for the tuning bench only; 0 means "use the measured table".
void launch_q5_volta_mma(const Tensor& x, const Weight& w, Tensor& out, bool add_residual,
                         std::int32_t weight_row_offset, WorkspaceArena& ws, cudaStream_t stream,
                         int splits_override = 0);
// Zero when the shape resolves to a single split; see the Q4 sibling.
[[nodiscard]] std::size_t q5_volta_mma_workspace_bytes(std::int32_t n, std::int32_t k,
                                                      std::int32_t t) noexcept;
[[nodiscard]] bool q5_volta_mma_supported(std::int32_t n, std::int32_t k,
                                          std::int32_t t) noexcept;
[[nodiscard]] int q5_volta_mma_splits(std::int32_t n, std::int32_t k, std::int32_t t) noexcept;
#endif

void launch_q5_gemv_r16_s2_x(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_simt_r4_c16(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_simt_split2_exact(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_q5_simt_split4_exact(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_q5_mma_r64_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_mma_r64_c128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
