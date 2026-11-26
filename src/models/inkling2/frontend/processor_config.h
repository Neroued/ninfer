#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace ninfer::models::inkling2 {

struct ProcessorConfig {
    std::uint32_t cell_height       = 40;
    std::uint32_t cell_width        = 40;
    std::uint32_t native_patch_size = 20;
    std::uint32_t merge_size        = 2;
    double rescale_factor           = 1.0 / 255.0;
    std::array<float, 3> mean{0.5F, 0.5F, 0.5F};
    std::array<float, 3> std{0.5F, 0.5F, 0.5F};
    std::string image_token = "<|unused_200054|>";

    [[nodiscard]] std::uint32_t features_per_patch() const noexcept {
        return 3U * native_patch_size * native_patch_size;
    }
};

ProcessorConfig parse_processor_config(std::string_view json);

} // namespace ninfer::models::inkling2
