#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "ops/linear/kernels/nvfp4/nvfp4_layout.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {

struct Nvfp4A4Workspace {
    std::uint8_t* codes  = nullptr;
    std::uint8_t* scales = nullptr;
    // Extent of the scale plane. The tiled layout writes over whole tiles, so it reaches past the
    // real token count; carrying the size lets the quantizer check that rather than trust that the
    // caller allocated through allocate_nvfp4_a4_workspace.
    std::size_t scale_bytes = 0;
    float* partials = nullptr;
    std::size_t partial_bytes = 0;
};

inline std::size_t nvfp4_a4_checked_bytes(std::int32_t tokens, std::size_t bytes_per_token) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 A4 workspace: T must be positive"); }
    const auto count = static_cast<std::size_t>(tokens);
    if (count > std::numeric_limits<std::size_t>::max() / bytes_per_token) {
        throw std::overflow_error("nvfp4 A4 workspace size overflow");
    }
    return count * bytes_per_token;
}

inline std::size_t nvfp4_a4_split_k_partial_bytes(std::int32_t rows, std::int32_t tokens,
                                                  std::int32_t partitions) {
    if (rows <= 0 || partitions <= 1)
        throw std::invalid_argument("NVFP4 split-K needs positive rows and multiple partitions");
    const auto width = static_cast<std::size_t>(rows) * partitions;
    if (width > std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::overflow_error("NVFP4 split-K partial row size overflow");
    return nvfp4_a4_checked_bytes(tokens, width * sizeof(float));
}

template <class Arena>
Nvfp4A4Workspace allocate_nvfp4_a4_workspace(Arena& arena, std::int32_t tokens,
                                             std::int32_t input_rows,
                                             std::size_t partial_bytes = 0) {
    if (input_rows <= 0 || (input_rows % 64) != 0) {
        throw std::invalid_argument("nvfp4 A4 workspace: invalid K");
    }
    const std::size_t code_bytes =
        nvfp4_a4_checked_bytes(tokens, static_cast<std::size_t>(input_rows) / 2);
    // The tiled layout addresses whole tiles, so the scale plane is allocated for the padded token
    // count: at most 255 tokens of scales, under 0.3 MiB on the widest registered K, and paid
    // whichever layout the quantizer then writes.
    const std::size_t scale_bytes = nvfp4_a4_checked_bytes(
        nvfp4_a4_padded_tokens(tokens), static_cast<std::size_t>(input_rows) / 16);
    const MemorySpan codes  = arena.alloc_bytes(code_bytes, 256);
    const MemorySpan scales = arena.alloc_bytes(scale_bytes, 256);
    auto* partials = partial_bytes
                         ? static_cast<float*>(arena.alloc_bytes(partial_bytes, 256).data)
                         : nullptr;
    return {static_cast<std::uint8_t*>(codes.data), static_cast<std::uint8_t*>(scales.data),
            scale_bytes, partials, partial_bytes};
}

inline std::size_t nvfp4_a4_workspace_capacity_bytes(std::int32_t tokens, std::int32_t input_rows,
                                                     std::size_t partial_bytes = 0) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_nvfp4_a4_workspace(layout, tokens, input_rows, partial_bytes);
    return layout.peak_bytes(1);
}

void launch_nvfp4_a4_quantize(const Tensor& x, const Weight& weight, Nvfp4A4Workspace workspace,
                              Nvfp4ScaleLayout layout, cudaStream_t stream);

} // namespace ninfer::ops::detail
