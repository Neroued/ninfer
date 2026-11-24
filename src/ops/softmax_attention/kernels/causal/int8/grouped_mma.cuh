#pragma once

#include "ops/softmax_attention/kernels/causal/ragged_grouped.cuh"
#include "ops/common/quant/integer.cuh"
#include "ops/kv_cache/kernels/hadamard.cuh"
#include "ops/kv_cache/kernels/layout.cuh"
#include "ops/softmax_attention/kernels/causal/tile_io.cuh"
#include "ops/softmax_attention/kernels/causal/operands.h"
#include "ops/softmax_attention/kernels/causal/int8/schedule.cuh"
#include "ops/softmax_attention/kernels/causal/partition.h"
#include "ops/softmax_attention/kernels/causal/epilogue.cuh"
#include "ops/softmax_attention/kernels/softmax.cuh"

namespace ninfer::ops::detail {

// Native INT8 QK accumulates each G64 group in INT32, then applies Q/K scales in FP32.
// PV dequantizes represented V to FP16 and accumulates in FP32.
#include "ops/softmax_attention/kernels/causal/int8/grouped_mma.inl"

#define NINFER_CAUSAL_RAGGED
#include "ops/softmax_attention/kernels/causal/int8/grouped_mma.inl"
#undef NINFER_CAUSAL_RAGGED

} // namespace ninfer::ops::detail
