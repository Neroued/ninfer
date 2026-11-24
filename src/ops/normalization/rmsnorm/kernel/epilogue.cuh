#pragma once

#include "ops/common/numeric/math.cuh"
#include "ops/normalization/rmsnorm/types.h"

namespace ninfer::ops::detail::rmsnorm {
template <Epilogue Kind>
__device__ __forceinline__ float apply_epilogue(float x, float inv, float weight, float z) {
    if constexpr (Kind == Epilogue::Offset) { weight += 1.0f; }
    float value = x * inv * weight;
    if constexpr (Kind == Epilogue::SiluGated) { value *= common::silu(z); }
    if constexpr (Kind == Epilogue::SigmoidGated) { value *= common::sigmoid(z); }
    return value;
}

} // namespace ninfer::ops::detail::rmsnorm
