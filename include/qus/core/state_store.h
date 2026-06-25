#pragma once

#include "qus/core/arena.h"
#include "qus/core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

namespace qus {

struct GdnState {
    std::vector<Tensor> conv;
    std::vector<Tensor> ssm;
    std::int32_t conv_dim       = 0;
    std::int32_t conv_width     = 0;
    std::int32_t value_heads    = 0;
    std::int32_t value_head_dim = 0;
    std::int32_t key_head_dim   = 0;
    DType conv_dtype            = DType::BF16;

    GdnState() = default;
    GdnState(DeviceArena& cache_arena, std::uint32_t gdn_layers, std::int32_t conv_dim,
             std::int32_t conv_width, std::int32_t value_heads, std::int32_t value_head_dim,
             std::int32_t key_head_dim, DType conv_dtype = DType::BF16);

    std::uint32_t layer_count() const noexcept;
    void reset(cudaStream_t stream = nullptr);
};

} // namespace qus
