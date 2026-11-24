#pragma once

#include "ops/softmax_attention/kernels/causal/ragged_grouped.cuh"
#include "ops/kv_cache/kernels/write/fp8.cuh"
#include "ops/common/quant/fp8.cuh"
#include "ops/kv_cache/kernels/layout.cuh"
#include "ops/softmax_attention/kernels/causal/tile_io.cuh"
#include "ops/softmax_attention/kernels/causal/operands.h"
#include "ops/softmax_attention/kernels/causal/fp8/schedule.cuh"
#include "ops/softmax_attention/kernels/causal/partition.h"
#include "ops/softmax_attention/kernels/causal/epilogue.cuh"
#include "ops/softmax_attention/kernels/softmax.cuh"
#include "ops/kv_cache/kernels/hadamard.cuh"

#include <type_traits>

namespace ninfer::ops::detail {

#include "ops/softmax_attention/kernels/causal/fp8/grouped_mma.inl"

#define NINFER_CAUSAL_RAGGED
#include "ops/softmax_attention/kernels/causal/fp8/grouped_mma.inl"
#undef NINFER_CAUSAL_RAGGED

} // namespace ninfer::ops::detail
