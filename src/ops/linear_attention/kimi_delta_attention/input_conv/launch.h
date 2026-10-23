#pragma once

#include <cuda_bf16.h>
#include "ninfer/ops/kda_input_proj.h"

namespace ninfer::ops::detail::kda_input {
inline constexpr int kProjectionFeatures = 12288, kInputFeatures = 4096;

struct Arguments {
    const float* weights;
    const __nv_bfloat16* source;
    __nv_bfloat16* destination;
    const std::int64_t *source_offsets, *destination_offsets;
    const std::int32_t* valid;
    __nv_bfloat16 *q, *k, *v, *records;
    int width, batch;
};

void launch(const Tensor&, const Weight&, const Arguments&, float* projected, cudaStream_t);
} // namespace ninfer::ops::detail::kda_input
