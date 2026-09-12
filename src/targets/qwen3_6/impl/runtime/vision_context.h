#pragma once

#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include <ninfer/targets/qwen3_6/vision_control.h>
#include "targets/qwen3_6/impl/runtime/layouts.h"
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
    static constexpr float attention_scale   = 0.11785113019775792F;
};

class VisionContext {
public:
    VisionContext(DeviceContext& device, const LoadedModelData& model);

    [[nodiscard]] static std::size_t workspace_bytes(const qwen3_6::VisionItemControl& item);
    [[nodiscard]] static std::size_t workspace_bytes(std::size_t patches,
                                                     std::size_t merged_tokens);
    [[nodiscard]] static VisionWorkspacePlan plan_workspace(std::uint32_t max_merged_tokens,
                                                            std::size_t general_capacity_bytes);
    [[nodiscard]] static Tensor bind_output(DeviceSpan backing, const VisionWorkspacePlan& plan,
                                            std::size_t merged_tokens);
    void encode(const VisionItemView& item, Tensor& output, DeviceSpan backing,
                const VisionWorkspacePlan& plan) const;

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
    VisionPrefillSession(DeviceContext& device, const LoadedModelData& model, DeviceSpan workspace,
                         const VisionWorkspacePlan& workspace_plan,
                         qwen3_6::PreparedPromptData& prompt, const VisionPrefillPlan& plan,
                         std::size_t& handoff_peak_bytes);

    [[nodiscard]] VisionChunk prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length);
    void release_encoded_media_payloads() noexcept;
    void retire_handoff() noexcept;
    [[nodiscard]] double elapsed_seconds() const;

    [[nodiscard]] std::size_t active_handoff_bytes() const noexcept {
        return active_handoff_bytes_;
    }

private:
    DeviceContext& device_;
    DeviceSpan workspace_;
    const VisionWorkspacePlan& workspace_plan_;
    qwen3_6::PreparedPromptData& prompt_;
    const VisionPrefillPlan& plan_;
    std::size_t& handoff_peak_bytes_;
    VisionContext context_;
    std::size_t next_use_ = 0;
    std::optional<std::uint32_t> active_item_;
    std::size_t active_handoff_bytes_ = 0;
    std::vector<std::uint32_t> encoded_payloads_pending_release_;
    std::vector<CudaEventTimer> timers_;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
