#pragma once

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::targets::qwen3_6 {

struct VisionItemControl {
    std::size_t patch_begin     = 0;
    std::size_t patch_count     = 0;
    std::size_t merged_begin    = 0;
    std::size_t merged_count    = 0;
    std::int32_t segment_length = 0;
    std::int32_t segment_count  = 0;
    std::size_t scatter_begin   = 0;
    std::size_t scatter_count   = 0;
};

struct VisionControl {
    std::vector<std::int32_t> position_ids;
    std::vector<std::int32_t> cu_seqlens;
    std::vector<std::int32_t> scatter_indices;
    std::vector<std::int32_t> position_table_indices;
    std::vector<float> position_table_weights;
    std::vector<VisionItemControl> items;
};

[[nodiscard]] VisionControl build_vision_control(const PreparedPromptData& prompt);

} // namespace ninfer::targets::qwen3_6
