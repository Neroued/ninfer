#pragma once

#include "models/common/qwen/frontend/prepared_prompt.h"
#include "models/common/qwen/vision/parameters.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::models::qwen {

struct VisionItemControl {
    models::frontend::Modality modality = models::frontend::Modality::Image;
    models::frontend::VisionGrid grid;
    std::size_t patch_begin     = 0;
    std::size_t patch_count     = 0;
    std::size_t merged_count    = 0;
    std::int32_t segment_length = 0;
    std::int32_t segment_count  = 0;
    std::vector<std::int32_t> position_ids;
    std::vector<std::int32_t> scatter_indices;
    std::vector<std::int32_t> position_table_indices;
    std::vector<float> position_table_weights;
};

struct VisionControl {
    std::uint32_t prepared_item_begin = 0;
    std::vector<VisionItemControl> items;
};

struct VisionItemControlPlan {
    std::uint32_t token_begin = 0;
    std::uint32_t token_end   = 0;
    std::size_t merged_count  = 0;
};

struct VisionControlPlan {
    std::int32_t spatial_merge_size = 0;
    std::int32_t position_grid_side = 0;
    std::vector<VisionItemControlPlan> items;
};

struct VisionUseSpan {
    std::uint32_t begin               = 0;
    std::uint32_t end                 = 0;
    std::uint32_t prepared_item_index = 0;
    std::uint32_t control_index       = 0;
};

struct VisionPrefillPlan {
    std::shared_ptr<const VisionControlPlan> control_plan;
    std::shared_ptr<const VisionControl> control;
    std::vector<VisionUseSpan> uses;
    std::size_t max_merged_count = 0;

    [[nodiscard]] std::uint32_t window_end(std::uint32_t begin, std::uint32_t end) const;
};

struct VisionExecutionInput {
    std::uint32_t prompt_tokens = 0;
    std::vector<models::frontend::VisionItem> vision_items;
    std::vector<std::shared_ptr<const models::frontend::PreparedMediaPayload>> media_payloads;
    VisionPrefillPlan plan;
};

[[nodiscard]] VisionControlPlan plan_vision_control(const PreparedPromptData& prompt,
                                                    const VisionConfig& config);
[[nodiscard]] VisionControl build_vision_control(const PreparedPromptData& prompt,
                                                 const VisionControlPlan& plan,
                                                 std::uint32_t prepared_item_begin);

} // namespace ninfer::models::qwen
