#pragma once

#include "qus/media/source.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace qus::media::internal {

struct Policy {
    std::size_t max_bytes                  = 256ULL << 20;
    std::uint64_t max_decoded_pixels       = 64ULL * 1024ULL * 1024ULL;
    std::uint64_t max_decoded_video_pixels = 128ULL * 1024ULL * 1024ULL;
    int max_video_source_frames            = 100'000;
    double max_video_duration_seconds      = 600.0;
    int connect_timeout_ms                 = 5'000;
    int timeout_ms                         = 60'000;
    int max_redirects                      = 3;
    bool allow_remote                      = true;
    bool allow_private_network             = false;
    std::filesystem::path media_root;
};

struct Image {
    int width  = 0;
    int height = 0;
    std::vector<std::uint8_t> rgb;
};

struct Video {
    int width        = 0;
    int height       = 0;
    int total_frames = 0;
    double fps       = 0.0;
    double duration  = 0.0;
    std::vector<int> indices;
    std::vector<Image> frames;
};

Image decode_image(const Source& source, const Policy& policy);
Video decode_video(const Source& source, const Policy& policy, double target_fps, int min_frames,
                   int max_frames);

} // namespace qus::media::internal
