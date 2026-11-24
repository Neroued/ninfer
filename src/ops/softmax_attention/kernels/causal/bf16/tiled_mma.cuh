#pragma once

#include "ops/softmax_attention/kernels/causal/ragged.h"

#include "ops/kv_cache/kernels/layout.cuh"
#include "ops/softmax_attention/kernels/causal/bf16/tile_io.cuh"
#include "ops/softmax_attention/kernels/softmax.cuh"
#include "ops/softmax_attention/kernels/causal/bf16/epilogue.cuh"
#include <math_constants.h>

namespace ninfer::ops::detail {

// Online attention, one CTA per (query tile, query head), with
// bottom-right causal alignment (query row i sees keys [0, base_pos + i]).
#include "ops/softmax_attention/kernels/causal/bf16/tiled_mma.inl"

#define NINFER_CAUSAL_RAGGED
#include "ops/softmax_attention/kernels/causal/bf16/tiled_mma.inl"
#undef NINFER_CAUSAL_RAGGED

} // namespace ninfer::ops::detail
