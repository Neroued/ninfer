#pragma once

#include "qus/core/arena.h"
#include "qus/core/device.h"
#include "qus/core/tensor.h"
#include "qus/core/weight.h"
#include "qus/core/weight_store.h"
#include "qus/model/config.h"
#include "qus/model/processor.h"

#include <array>
#include <cstddef>

namespace qus::model {

enum class VisionTapId {
    PatchEmbed,
    Block,
    Merger,
};

using VisionTapCallback = void (*)(void*, VisionTapId, int, const Tensor&, cudaStream_t);

// Fixed Qwen3.6-27B Vision tower. encode() resets `workspace` and returns a
// [5120,V] BF16 tensor whose storage remains valid until that workspace is reset.
class Qwen3_6_Vision {
public:
    Qwen3_6_Vision(DeviceContext& ctx, const WeightStore& weights);

    [[nodiscard]] static std::size_t workspace_bytes(const ProcessedInput& input);
    [[nodiscard]] Tensor encode(const ProcessedInput& input, WorkspaceArena& workspace,
                                void* tap = nullptr, VisionTapCallback callback = nullptr) const;

private:
    struct BlockW {
        const Tensor* norm1_weight    = nullptr;
        const Tensor* norm1_bias      = nullptr;
        const Weight* qkv             = nullptr;
        const Tensor* qkv_bias        = nullptr;
        const Weight* projection      = nullptr;
        const Tensor* projection_bias = nullptr;
        const Tensor* norm2_weight    = nullptr;
        const Tensor* norm2_bias      = nullptr;
        const Weight* fc1             = nullptr;
        const Tensor* fc1_bias        = nullptr;
        const Weight* fc2             = nullptr;
        const Tensor* fc2_bias        = nullptr;
    };

    struct MergerW {
        const Tensor* norm_weight = nullptr;
        const Tensor* norm_bias   = nullptr;
        const Weight* fc1         = nullptr;
        const Tensor* fc1_bias    = nullptr;
        const Weight* fc2         = nullptr;
        const Tensor* fc2_bias    = nullptr;
    };

    DeviceContext& ctx_;
    const Weight* patch_embed_      = nullptr;
    const Tensor* patch_embed_bias_ = nullptr;
    const Tensor* position_embed_   = nullptr;
    std::array<BlockW, VisionConfig::depth> blocks_{};
    MergerW merger_{};
};

} // namespace qus::model
