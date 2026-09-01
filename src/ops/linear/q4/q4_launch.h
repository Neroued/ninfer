#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

using Q4Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

#ifdef NINFER_VOLTA_BUILD
// Fused-dequant tensor-core route (see q4_volta_mma_gemm.cuh). Kept out of the Q4Launch table
// deliberately: split-K needs an fp32 accumulation workspace, which that signature cannot carry.
// `weight_row_offset` selects a contiguous row band of a parent weight; see the Q5 sibling.
// `splits_override` is for the tuning bench only; 0 means "use the measured table".
void launch_q4_volta_mma(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                         cudaStream_t stream, std::int32_t weight_row_offset = 0,
                         int splits_override = 0);
// Zero when the shape resolves to a single split: that path stores BF16 straight to `out` and
// allocates no fp32 accumulator at all.
[[nodiscard]] std::size_t q4_volta_mma_workspace_bytes(std::int32_t n, std::int32_t k,
                                                      std::int32_t t) noexcept;
[[nodiscard]] bool q4_volta_mma_supported(std::int32_t n, std::int32_t k,
                                          std::int32_t t) noexcept;
[[nodiscard]] int q4_volta_mma_splits(std::int32_t n, std::int32_t k, std::int32_t t) noexcept;

// Quadpair-split-N form of the same instruction (q4_volta_qpn_gemm.cuh), for the narrow T the
// companion kernel wastes its A rows on. No split-K workspace: the CTA's warps split K and
// reduce in shared memory, so this fits a plain launcher signature.
void launch_q4_volta_qpn(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream,
                         std::int32_t weight_row_offset = 0);
[[nodiscard]] bool q4_volta_qpn_supported(std::int32_t n, std::int32_t k,
                                          std::int32_t t) noexcept;

// Band for the quadpair-split-N route (q4_volta_qpn_gemm.cuh), measured at k=5120 against
// whichever route production would otherwise take -- sliced SIMT below T=9, the fused 32x8 kernel
// from 9 up. Microseconds, incumbent -> qpn:
//
//   T:            4              5              6              7              8            9
//   n= 1024   22.5 / 55.3    33.8 / 60.4    35.8 / 60.4    36.9 / 61.4    30.7 / 54.3    30.7/88.1
//   n= 4096   51.1 / 66.6    89.1 / 71.7    94.2 / 71.7   103.4 / 71.7    73.7 / 66.6   77.8/100.4
//   n= 6144   64.6 / 74.8   110.6 / 77.7   118.8 / 78.8   128.0 / 78.8    95.2 / 75.9  102.4/114.7
//   n= 7168   73.6 / 77.8   123.9 / 80.9   133.1 / 80.9   142.4 / 80.9   107.6 / 78.8  135.1/120.8
//   n=34816  284.7 /298.0   476.2 /300.0   512.0 /302.1   567.3 /306.2   452.6 /309.2  539.6/531.5
//
// Both edges belong to the incumbent, not to this kernel -- QPN is flat in T, as the mapping
// intends. T=4 is exactly one eight-column SIMT tile and is genuinely efficient; from T=5 SIMT
// pays a second, mostly empty tile and loses by up to 1.85x. At T=9 QPN needs a second 8-row A
// tile, which nearly doubles its cost, while the 32x8 kernel absorbs those rows for free.
//
// n=1024 is excluded because 32 CTAs cannot fill the machine. There is no upper n bound: the
// per-lane weight streams that used to thrash L2 at n=34816 are fixed inside the kernel by
// consuming a whole cache line per lane per iteration.
constexpr std::int32_t kVoltaQpnMinT    = 5;
constexpr std::int32_t kVoltaQpnMaxT    = 8;
constexpr std::int32_t kVoltaQpnMinRows = 4096;

[[nodiscard]] inline bool q4_uses_volta_qpn(std::int32_t n, std::int32_t k,
                                            std::int32_t t) noexcept {
    return t >= kVoltaQpnMinT && t <= kVoltaQpnMaxT && n >= kVoltaQpnMinRows &&
           q4_volta_qpn_supported(n, k, t);
}

#endif

void launch_q4_gemv_r4_w1_direct(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_q4_gemv_r1_w8_direct(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_q4_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_draft_head_small_t(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream);
void launch_q4_mma_r64_c32(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c48(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c56(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c64_endpoint(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream);
void launch_q4_mma_r64_c72(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c80(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c96(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c104_bounded(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream);
void launch_q4_mma_r64_c112_partial(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream);
void launch_q4_mma_r64_c112(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c120_partial(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream);
void launch_q4_mma_r64_c120(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
