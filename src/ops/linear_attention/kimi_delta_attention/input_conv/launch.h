#pragma once

#include <cuda_bf16.h>
#include "core/layout.h"
#include "ninfer/ops/kda_input_proj.h"

namespace ninfer::ops::detail::kda_input {
inline constexpr int kProjectionFeatures = 12288, kInputFeatures = 4096;

struct Arguments {
    const void* weights;
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

struct Mxfp8Workspace {
    LayoutRegion codes, scales, convolution;
    std::size_t bytes;
};

inline Mxfp8Workspace mxfp8_workspace(int width, int batch, bool update, LinearPolicy policy,
                                      bool boundaries) {
    LayoutBuilder layout;
    Mxfp8Workspace out{};
    const auto columns = std::size_t(width) * batch;
    if (policy == LinearPolicy::AllowA8) {
        out.codes  = layout.add(columns * kInputFeatures, 256, "MXFP8 activation codes");
        out.scales = layout.add(columns * (kInputFeatures / 32), 256, "MXFP8 activation scales");
    }
    if (boundaries)
        out.convolution = layout.add(tiled_workspace(width, batch, update).bytes, 256,
                                     "KDA convolution boundaries");
    out.bytes = layout.finish(1);
    return out;
}

std::size_t workspace_capacity(int batch, int first, int last, bool update);

void launch(const Tensor&, const Weight&, const Arguments&, float* scratch, cudaStream_t);

std::size_t mxfp8_workspace_capacity(int batch, int first, int last, bool update,
                                     LinearPolicy policy);
Mxfp8Workspace mxfp8_execution_workspace(int width, int batch, bool update, LinearPolicy policy);
void launch_mxfp8(const Tensor&, const Weight&, LinearPolicy, const Arguments&, void* scratch,
                  cudaStream_t);
} // namespace ninfer::ops::detail::kda_input
