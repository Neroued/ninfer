#pragma once

#include "ops/softmax_attention/kernels/causal/ragged_grouped.cuh"
#include "ops/kv_cache/kernels/write/k8v4.cuh"
#include "ops/kv_cache/kernels/hadamard.cuh"
#include "ops/kv_cache/kernels/layout.cuh"
#include "ops/softmax_attention/kernels/causal/operands.h"
#include "ops/common/quant/fp8.cuh"
#include "ops/common/quant/nvfp4.cuh"
#include "ops/softmax_attention/kernels/causal/k8v4/schedule.cuh"
#include "ops/softmax_attention/kernels/causal/partition.h"
#include "ops/softmax_attention/kernels/causal/epilogue.cuh"
#include "ops/softmax_attention/kernels/softmax.cuh"

namespace ninfer::ops::detail {
// Rotated FP8 Q/K use native FP8 MMA. Rotated NVFP4 V widens to FP16 for FP32 PV accumulation.
#include "ops/softmax_attention/kernels/causal/k8v4/grouped_mma.inl"

#define NINFER_CAUSAL_RAGGED
#include "ops/softmax_attention/kernels/causal/k8v4/grouped_mma.inl"
#undef NINFER_CAUSAL_RAGGED

} // namespace ninfer::ops::detail
