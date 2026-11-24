#pragma once

#include "ops/softmax_attention/kernels/causal/ragged.h"

#include "ops/kv_cache/kernels/layout.cuh"
#include "ops/common/quant/fp8.cuh"
#include "ops/common/quant/nvfp4.cuh"
#include "ops/kv_cache/kernels/hadamard.cuh"
#include "ops/softmax_attention/kernels/causal/epilogue.cuh"
#include "ops/softmax_attention/kernels/causal/operands.h"
#include "ops/softmax_attention/kernels/causal/partition.h"
#include "ops/softmax_attention/kernels/softmax.cuh"
#include "ops/softmax_attention/kernels/causal/tile_io.cuh"

namespace ninfer::ops::detail {

struct Fp8KvTiledValues {
    using Scale                            = std::uint8_t;
    using SharedScale                      = unsigned;
    static constexpr int kCodeBytes        = 256;
    static constexpr int kScaleItems       = 2;
    static constexpr bool kInverseRotation = false;

    template <bool SingleFactor>
    __device__ __forceinline__ static int4 expand(const std::uint8_t* codes, SharedScale scale) {
        if constexpr (SingleFactor)
            return common::fp8_dequant_f16x8_scaled(codes, __ushort_as_half(scale));
        else
            return common::fp8_dequant_f16x8(codes, scale >> 16);
    }
};

struct K8V4KvTiledValues {
    using Scale                            = std::uint8_t;
    using SharedScale                      = Scale;
    static constexpr int kCodeBytes        = 128;
    static constexpr int kScaleItems       = 16;
    static constexpr bool kInverseRotation = true;

    template <bool SingleFactor>
    __device__ __forceinline__ static int4 expand(const std::uint8_t* codes, Scale scale) {
        return common::nvfp4_dequant_f16x8(codes, scale);
    }
};

// Each warp retains 16 query rows through MXFP8 QK, softmax, and FP16 PV.
// Values selects the stored V representation; split outputs remain FP32 with
// maxima in natural scaled-score units for the final merge.
#include "ops/softmax_attention/kernels/causal/mxfp8_tiled_mma.inl"

#define NINFER_CAUSAL_RAGGED
#include "ops/softmax_attention/kernels/causal/mxfp8_tiled_mma.inl"
#undef NINFER_CAUSAL_RAGGED

} // namespace ninfer::ops::detail
