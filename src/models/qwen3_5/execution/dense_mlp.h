#pragma once

#include "core/arena.h"
#include "models/qwen3_5/execution/profile.h"
#include "core/device.h"
#include "ninfer/ops/projection_weights.h"

namespace ninfer::models::qwen3_5::execution {

struct DenseParameters {
    ops::SingleProjectionWeight gate_up;
    ops::SingleProjectionWeight down;
};

enum class DenseMlpUse { Text, Mtp };

[[nodiscard]] std::size_t dense_mlp_workspace_bytes(const DenseProfile& parameters,
                                                    std::int32_t first, std::int32_t last,
                                                    DenseMlpUse use);
void dense_mlp(const Tensor& hidden, const DenseParameters& parameters, Tensor& residual,
               DenseMlpUse use, MemoryArena& workspace, CudaExecutionView execution);

} // namespace ninfer::models::qwen3_5::execution
