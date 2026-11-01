#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace ninfer::models::glm5next {

struct MediaProcessorConfig {
    std::uint32_t patch_size          = 14;
    std::uint32_t temporal_patch_size = 2;
    std::uint32_t merge_size          = 2;
    std::uint32_t patch_expand_factor = 1;
    std::uint32_t min_tokens          = 16;
    std::uint32_t max_tokens          = 8000;
    std::array<float, 3> mean{0.48145466F, 0.4578275F, 0.40821073F};
    std::array<float, 3> std{0.26862954F, 0.26130258F, 0.27577711F};

    [[nodiscard]] std::uint32_t features_per_patch() const noexcept {
        return 3 * temporal_patch_size * patch_size * patch_size;
    }
};

struct ProcessorConfig {
    MediaProcessorConfig image;
    MediaProcessorConfig video;
    double video_fps               = 2;
    std::uint32_t max_video_frames = 2048;
};

ProcessorConfig parse_processor_config(std::string_view json);

} // namespace ninfer::models::glm5next
