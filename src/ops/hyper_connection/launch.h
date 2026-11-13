#pragma once

// ninfer::ops::detail - private launch prototypes of the hyper-connection family.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// The mixer geometry compiled into the fused route (4 streams of 2560, low rank 320).
inline constexpr std::int32_t kHcStreams   = 4;
inline constexpr std::int32_t kHcHidden    = 2560;
inline constexpr std::int32_t kHcWide      = kHcStreams * kHcHidden;
inline constexpr std::int32_t kHcRank      = 320;
inline constexpr std::int32_t kHcRowsCap   = kHcRank + kHcStreams;
inline constexpr std::int32_t kHcFusedMaxT = 8;

void grouped_offset_rmsnorm_launch(const Tensor& x, const Tensor& weight, std::int32_t groups,
                                   float eps, Tensor& out, cudaStream_t stream);
// projection [rows,T] -> lowrank [lowrank_rows,T] BF16 = silu(p/S); inject [S,T] FP32 = 2
// sigmoid(p/S)
void hyper_connection_gates_launch(const Tensor& projection, std::int32_t streams, Tensor& lowrank,
                                   Tensor* inject, cudaStream_t stream);
// up_collapse.cu: the general route's Up projection fused with the collapse for a Q8
// Up at T > 128 (bit for bit the unfused Linear + collapse). force_for_test: -1 automatic, 0 never,
// 1 whenever supported.
[[nodiscard]] bool hc_up_collapse_supported(const Weight& up, std::int32_t hidden,
                                            std::int32_t tokens);
void hc_up_collapse_launch(const Tensor& lowrank, const Weight& up, const Tensor& normalized,
                           Tensor& mixed, cudaStream_t stream);
void hc_up_collapse_force_for_test(int mode);

void hyper_connection_collapse_launch(const Tensor& up, const Tensor& normalized,
                                      std::int32_t streams, Tensor& mixed, cudaStream_t stream);
void hyper_connection_combine_launch(const Tensor& y, const Tensor& inject, Tensor& residual,
                                     cudaStream_t stream);
void hyper_connection_expand_launch(const Tensor& x, std::int32_t streams, Tensor& residual,
                                    cudaStream_t stream);

// Small-T mix (T <= 8) over Q8 row-split, NVFP4 or BF16 Down/Up: two launches (mix_small_t.cu).
// Compiled for H = kHcHidden (every format) and for Q8 at H = 2 kHcHidden.
// The residual is written only with a pending combine (previous_y non-null).
struct HcFusedMix {
    __nv_bfloat16* residual         = nullptr; // [S*H,T]
    const __nv_bfloat16* previous_y = nullptr; // [H,T] or nullptr (no pending combine)
    const float* previous_inject    = nullptr; // [S,T]
    const __nv_bfloat16* norm       = nullptr; // [S*H]
    // Q8 [rows, S*H] / NVFP4 E2M1 [padded rows, S*H/2] / BF16 [rows, S*H] (bytes)
    const std::int8_t* down_codes = nullptr;
    const __half* down_scales     = nullptr; // Q8 [rows, S*H/32]
    // Q8 [S*H, 384] / NVFP4 E2M1 [S*H, R/2] / BF16 [S*H, R] (bytes)
    const std::int8_t* up_codes = nullptr;
    const __half* up_scales     = nullptr; // Q8 [S*H, 12]
    __nv_bfloat16* mixed        = nullptr; // [H,T]
    float* inject               = nullptr; // [S,T] or nullptr
    float* square_sums          = nullptr; // workspace [T, S]
    __half* low                 = nullptr; // workspace [T, R] low-rank activation (FP16; BF16
                                           // bits with BF16 weights)
    std::int32_t rows   = 0;               // kHcRank or kHcRowsCap
    std::int32_t tokens = 0;
    float eps           = 0.0F;
    // NVFP4 weights (option): E4M3 scale planes (BlockScaleK16M128x4) and matrix divisors; Down is
    // stored with `down_padded_rows` (a multiple of 128) rows.
    bool fp4                            = false;
    const std::uint8_t* down_fp4_scales = nullptr;
    const std::uint8_t* up_fp4_scales   = nullptr;
    float down_divisor                  = 1.0F;
    float up_divisor                    = 1.0F;
    std::int32_t down_padded_rows       = 0;
    // BF16 weights (the FP4-Local checkpoint's mixers): no scales, BF16 activation operands.
    bool bf16 = false;
    // Hidden width H: kHcHidden, or 2 kHcHidden for Q8 (hyper_connection_fused_compiled).
    std::int32_t hidden = kHcHidden;
};

// TMA descriptors of the small-T route (encoded per launch from HcFusedMix's weight pointers).
struct HcFusedMaps {
    CUtensorMap down_codes;
    CUtensorMap down_scales;
    CUtensorMap up_codes;
};

// Whether the current device runs the small-T route (thread block clusters, sm_90+).
[[nodiscard]] bool hyper_connection_fused_supported();
// Whether the small-T route is compiled for hidden width H and the weight format (Q8 when neither
// fp4 nor bf16): every format at kHcHidden, Q8 also at 2 kHcHidden.
[[nodiscard]] bool hyper_connection_fused_compiled(std::int32_t hidden, bool fp4, bool bf16);
[[nodiscard]] std::size_t hyper_connection_fused_workspace_bytes(std::int32_t tokens);
void hyper_connection_fused_mix_launch(const HcFusedMix& params, cudaStream_t stream);

} // namespace ninfer::ops::detail
