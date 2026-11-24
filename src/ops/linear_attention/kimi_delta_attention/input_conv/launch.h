#pragma once

#include <cuda_bf16.h>
#include <algorithm>
#include "core/layout.h"
#include "core/platform.h"
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
    // For ragged calls, width is the total packed token count and batch is the segment count.
    const std::int32_t *cu_seqlens = nullptr, *seq_lengths = nullptr, *token_to_seq = nullptr;
};

inline constexpr int kTiledTokens = 128;

inline bool whole_window_tiles(int width) {
    return width <= kTiledTokens && kTiledTokens % width == 0;
}

struct TiledWorkspace {
    std::size_t edge_words;
    std::size_t bytes;
};

inline TiledWorkspace tiled_workspace(std::int64_t columns, std::int64_t tail_columns) {
    const auto edges =
        std::size_t((columns + kTiledTokens - 1) / kTiledTokens) * 6 * kProjectionFeatures;
    return {edges, edges * sizeof(float) +
                       std::size_t(tail_columns) * kProjectionFeatures * sizeof(__nv_bfloat16)};
}

inline TiledWorkspace tiled_workspace(int width, int batch, bool update) {
    return tiled_workspace(std::int64_t(width) * batch, update ? 3LL * batch : 0);
}

inline TiledWorkspace ragged_tiled_workspace(int tokens, int segments) {
    return tiled_workspace(tokens, std::min<std::int64_t>(tokens, 3LL * segments));
}

inline constexpr int kQ8StagedTokenLimit = 32;

inline bool ragged_q8_staged(int tokens) {
    return !NINFER_TARGET_SPARK || tokens <= kQ8StagedTokenLimit;
}

inline std::size_t ragged_q8_workspace(int tokens, int segments) {
    return ragged_q8_staged(tokens) ? std::size_t(tokens) * kProjectionFeatures * sizeof(float)
                                    : ragged_tiled_workspace(tokens, segments).bytes;
}

// The token interval is feasible for packed execution. The segment ceiling is feasible at last.
inline std::size_t ragged_q8_workspace_capacity(int first, int last, int max_segments) {
    auto peak = ragged_q8_workspace(last, max_segments);
    if (ragged_q8_staged(first) && !ragged_q8_staged(last)) {
        const int end = kQ8StagedTokenLimit;
        peak          = std::max(peak, ragged_q8_workspace(end, std::min(max_segments, end)));
    }
    return peak;
}

struct Mxfp8Workspace {
    LayoutRegion codes, scales, convolution;
    std::size_t bytes;
};

inline Mxfp8Workspace mxfp8_workspace(std::size_t columns, LinearPolicy policy,
                                      std::size_t convolution_bytes) {
    LayoutBuilder layout;
    Mxfp8Workspace out{};
    if (policy == LinearPolicy::AllowA8) {
        out.codes  = layout.add(columns * kInputFeatures, 256, "MXFP8 activation codes");
        out.scales = layout.add(columns * (kInputFeatures / 32), 256, "MXFP8 activation scales");
    }
    if (convolution_bytes)
        out.convolution = layout.add(convolution_bytes, 256, "KDA convolution boundaries");
    out.bytes = layout.finish(1);
    return out;
}

inline Mxfp8Workspace mxfp8_workspace(int width, int batch, bool update, LinearPolicy policy,
                                      bool boundaries) {
    return mxfp8_workspace(std::size_t(width) * batch, policy,
                           boundaries ? tiled_workspace(width, batch, update).bytes : 0);
}

inline Mxfp8Workspace ragged_mxfp8_workspace(int tokens, int segments, LinearPolicy policy) {
    return mxfp8_workspace(tokens, policy, ragged_tiled_workspace(tokens, segments).bytes);
}

std::size_t workspace_capacity(int batch, int first, int last, bool update);

void launch(const Tensor&, const Weight&, const Arguments&, float* scratch, cudaStream_t);

std::size_t mxfp8_workspace_capacity(int batch, int first, int last, bool update,
                                     LinearPolicy policy);
Mxfp8Workspace mxfp8_execution_workspace(int width, int batch, bool update, LinearPolicy policy);
void launch_mxfp8(const Tensor&, const Weight&, LinearPolicy, const Arguments&, void* scratch,
                  cudaStream_t);
} // namespace ninfer::ops::detail::kda_input
