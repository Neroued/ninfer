#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::media::decode {

enum class ErrorKind {
    BudgetExceeded,
    InvalidInput,
};

class Error final : public std::runtime_error {
public:
    Error(ErrorKind kind, std::string message)
        : std::runtime_error(std::move(message)), kind_(kind) {}

    [[nodiscard]] ErrorKind kind() const noexcept { return kind_; }

private:
    ErrorKind kind_;
};

struct Policy {
    std::size_t max_bytes                  = 256ULL << 20;
    std::uint64_t max_decoded_pixels       = 64ULL * 1024ULL * 1024ULL;
    std::uint64_t max_decoded_video_pixels = 128ULL * 1024ULL * 1024ULL;
    int max_video_source_frames            = 100'000;
    double max_video_duration_seconds      = 600.0;
    std::function<void()> checkpoint;
};

struct Image {
    int width  = 0;
    int height = 0;
    std::vector<std::uint8_t> rgb;
};

struct ImageInfo {
    int width  = 0;
    int height = 0;
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

struct VideoInfo {
    int width          = 0;
    int height         = 0;
    int total_frames   = 0;
    int sampled_frames = 0;
    double fps         = 0.0;
    double duration    = 0.0;
    std::vector<int> indices;
};

struct VideoSource {
    int total_frames           = 0;
    double fps                 = 0.0;
    double duration            = 0.0;
    std::uint64_t coded_pixels = 0;
};

// A model selects unique source frames. A truncated tail is permitted only after minimum_frames.
// Repetition and temporal padding belong to the model processor.
struct VideoSelection {
    VideoSource source;
    std::vector<int> indices;
    int minimum_frames = 1;
};

ImageInfo inspect_image(std::span<const std::uint8_t> bytes, const Policy& policy);
VideoSource probe_video(std::span<const std::uint8_t> bytes, const Policy& policy);
VideoInfo inspect_video(std::span<const std::uint8_t> bytes, const Policy& policy,
                        const VideoSelection& selection);
Image decode_image(std::span<const std::uint8_t> bytes, const Policy& policy);
Video decode_video(std::span<const std::uint8_t> bytes, const Policy& policy,
                   const VideoSelection& selection);

} // namespace ninfer::media::decode
