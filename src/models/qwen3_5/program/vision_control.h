#pragma once

#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/config.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_5 {

struct VisionItemControl {
    PromptModality modality = PromptModality::Image;
    VisionGrid grid;
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

[[nodiscard]] VisionControlPlan plan_vision_control(const PreparedPromptData& prompt,
                                                    const VisionConfig& config);
[[nodiscard]] VisionControl build_vision_control(const PreparedPromptData& prompt,
                                                 const VisionControlPlan& plan,
                                                 std::uint32_t prepared_item_begin);

} // namespace ninfer::models::qwen3_5
