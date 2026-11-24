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

inline constexpr int kTiledTokens = 128;

struct TiledWorkspace {
    std::size_t edge_words;
    std::size_t bytes;
};

inline TiledWorkspace tiled_workspace(int width, int batch, bool update) {
    const auto columns = std::int64_t(width) * batch;
    const auto edges =
        std::size_t((columns + kTiledTokens - 1) / kTiledTokens) * 6 * kProjectionFeatures;
    return {edges, edges * sizeof(float) + (update ? std::size_t(batch) * 3 * kProjectionFeatures *
                                                         sizeof(__nv_bfloat16)
                                                   : 0)};
}

std::size_t workspace_capacity(int batch, int first, int last, bool update);

void launch(const Tensor&, const Weight&, const Arguments&, float* scratch, cudaStream_t);
} // namespace ninfer::ops::detail::kda_input
