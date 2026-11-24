#pragma once

#include "ops/softmax_attention/kernels/causal/ragged_grouped.cuh"
#include "ops/common/quant/nvfp4.cuh"
#include "ops/kv_cache/kernels/hadamard.cuh"
#include "ops/kv_cache/kernels/layout.cuh"
#include "ops/softmax_attention/kernels/causal/tile_io.cuh"
#include "ops/softmax_attention/kernels/causal/operands.h"
#include "ops/softmax_attention/kernels/causal/nvfp4/schedule.cuh"
#include "ops/softmax_attention/kernels/causal/partition.h"
#include "ops/softmax_attention/kernels/causal/epilogue.cuh"
#include "ops/softmax_attention/kernels/softmax.cuh"

namespace ninfer::ops::detail {
// Decode represented G16 cache to FP16. Q remains unquantized; QK/PV accumulate in FP32.
#include "ops/softmax_attention/kernels/causal/nvfp4/grouped_mma.inl"

#define NINFER_CAUSAL_RAGGED
#include "ops/softmax_attention/kernels/causal/nvfp4/grouped_mma.inl"
#undef NINFER_CAUSAL_RAGGED

} // namespace ninfer::ops::detail
