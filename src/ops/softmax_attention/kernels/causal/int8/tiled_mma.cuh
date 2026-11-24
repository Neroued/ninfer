#pragma once

#include "ops/softmax_attention/kernels/causal/ragged.h"
#include "ops/common/quant/integer.cuh"
#include "ops/kv_cache/kernels/hadamard.cuh"
#include "ops/kv_cache/kernels/layout.cuh"
#include "ops/softmax_attention/kernels/causal/tile_io.cuh"
#include "ops/softmax_attention/kernels/causal/operands.h"
#include "ops/softmax_attention/kernels/causal/int8/schedule.cuh"
#include "ops/softmax_attention/kernels/causal/epilogue.cuh"
#include "ops/softmax_attention/kernels/softmax.cuh"

namespace ninfer::ops::detail {

#include "ops/softmax_attention/kernels/causal/int8/tiled_mma.inl"

#define NINFER_CAUSAL_RAGGED
#include "ops/softmax_attention/kernels/causal/int8/tiled_mma.inl"
#undef NINFER_CAUSAL_RAGGED

} // namespace ninfer::ops::detail
