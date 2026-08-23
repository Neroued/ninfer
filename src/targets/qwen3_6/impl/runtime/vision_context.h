#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include <ninfer/targets/qwen3_6/vision_control.h>
#include "runtime/contract/transient_region.h"
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

struct VisionItemView {
    std::span<const std::uint16_t> patches;
    const qwen3_6::VisionItemControl* control = nullptr;
};

struct VisionScheduleConfig {
    static constexpr int layers              = VisionConfig::layers;
    static constexpr int hidden              = VisionConfig::hidden;
    static constexpr int intermediate        = VisionConfig::intermediate;
    static constexpr int out_hidden          = VisionConfig::output_hidden;
    static constexpr int heads               = VisionConfig::heads;
    static constexpr int head_dim            = VisionConfig::head_dim;
    static constexpr int patch_dim           = VisionConfig::patch_dim;
    static constexpr int merge_unit          = VisionConfig::merge_unit;
    static constexpr int merger_hidden       = VisionConfig::merger_hidden;
    static constexpr int position_embeddings = VisionConfig::position_embeddings;
    static constexpr int rotary_dim          = VisionConfig::rotary_dim;
    static constexpr float rope_theta        = VisionConfig::rope_theta;
    static constexpr float norm_eps          = VisionConfig::norm_epsilon;
};

class VisionContext {
public:
    VisionContext(DeviceContext& device, const LoadedModelData& model);

    [[nodiscard]] static std::size_t output_transient_bytes(std::size_t merged_tokens);
    [[nodiscard]] static std::size_t workspace_bytes(const qwen3_6::VisionItemControl& item);
    [[nodiscard]] static std::size_t workspace_capacity_bytes(std::uint32_t max_merged_tokens,
                                                              std::uint32_t max_segments);
    void encode(const VisionItemView& item, Tensor& output, WorkspaceArena& workspace) const;

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
    std::array<BlockW, VisionScheduleConfig::layers> blocks_{};
    MergerW merger_{};
};

struct VisionChunk {
    std::int32_t length                       = 0;
    const qwen3_6::VisionItemControl* control = nullptr;
    Tensor embeddings;
};

class VisionPrefillSession {
public:
    VisionPrefillSession(DeviceContext& device, const LoadedModelData& model,
                         WorkspaceArena& workspace, qwen3_6::PreparedPromptData& prompt,
                         const VisionPrefillPlan& plan, runtime::TransientRegion transient);

    [[nodiscard]] VisionChunk prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length);
    void release_encoded_media_payloads() noexcept;
    [[nodiscard]] double elapsed_seconds() const;

private:
    DeviceContext& device_;
    WorkspaceArena& workspace_;
    qwen3_6::PreparedPromptData& prompt_;
    const VisionPrefillPlan& plan_;
    runtime::TransientRegion transient_;
    VisionContext context_;
    std::optional<std::uint32_t> active_item_;
    std::vector<std::uint32_t> encoded_payloads_pending_release_;
    std::vector<CudaEventTimer> timers_;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
