#pragma once

#include "core/arena.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "ops/linear/nvfp4/nvfp4_config.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {

struct Nvfp4W4a4Workspace {
    std::uint8_t* codes          = nullptr;
    std::uint8_t* scales         = nullptr;
    void* tma_descriptor_storage = nullptr;
};

#ifdef _WIN32
inline constexpr std::size_t kNvfp4W4a4TmaDescriptorBytes = 4 * 128;
#endif

inline std::size_t nvfp4_w4a4_checked_bytes(std::int32_t tokens, std::size_t bytes_per_token) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 W4A4 workspace: T must be positive"); }
    const auto count = static_cast<std::size_t>(tokens);
    if (count > std::numeric_limits<std::size_t>::max() / bytes_per_token) {
        throw std::overflow_error("nvfp4 W4A4 workspace size overflow");
    }
    return count * bytes_per_token;
}

template <class Arena>
Nvfp4W4a4Workspace allocate_nvfp4_w4a4_workspace(Arena& arena, std::int32_t tokens,
                                                 std::int32_t input_rows) {
    if (input_rows <= 0 || (input_rows % 64) != 0) {
        throw std::invalid_argument("nvfp4 W4A4 workspace: invalid K");
    }
    const std::size_t code_bytes =
        nvfp4_w4a4_checked_bytes(tokens, static_cast<std::size_t>(input_rows) / 2);
    const std::size_t scale_bytes =
        nvfp4_w4a4_checked_bytes(tokens, static_cast<std::size_t>(input_rows) / 16);
    const DeviceSpan codes  = arena.alloc_bytes(code_bytes, 256);
    const DeviceSpan scales = arena.alloc_bytes(scale_bytes, 256);
#ifdef _WIN32
    const DeviceSpan tma_descriptors = arena.alloc_bytes(kNvfp4W4a4TmaDescriptorBytes, 128);
    return {static_cast<std::uint8_t*>(codes.data), static_cast<std::uint8_t*>(scales.data),
            tma_descriptors.data};
#else
    return {static_cast<std::uint8_t*>(codes.data), static_cast<std::uint8_t*>(scales.data), nullptr};
#endif
}

inline std::size_t nvfp4_w4a4_workspace_capacity_bytes(std::int32_t tokens,
                                                       std::int32_t input_rows) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_nvfp4_w4a4_workspace(layout, tokens, input_rows);
    return layout.peak_bytes(1);
}

void launch_nvfp4_w4a4_quantize(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                                cudaStream_t stream);

void launch_nvfp4_w4a4(const Tensor& x, const Weight& weight, Tensor& out,
                       Nvfp4W4a4Workspace workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
