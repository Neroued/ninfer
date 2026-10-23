#pragma once

#include "models/common/frontend/resources.h"

#include <memory>
#include <optional>
#include <string_view>

namespace ninfer::models::qwen {

inline constexpr std::size_t kPreparedVisionPatchFeatures = 3ULL * 2ULL * 16ULL * 16ULL;
inline constexpr std::uint64_t kRawPatchesPerVisionToken  = 4;
// Aggregate prompt capacity and one-item execution capacity are intentionally distinct. Multiple
// media items are retained by one prepared prompt but pass through the Vision tower sequentially.
inline constexpr std::uint64_t kMaximumPromptVisionTokens = 32'768;
inline constexpr std::uint64_t kMaximumPromptVisionRawPatches =
    kMaximumPromptVisionTokens * kRawPatchesPerVisionToken;
inline constexpr std::uint64_t kMaximumVisionItemTokens = 16'384;
inline constexpr std::uint64_t kMaximumVisionItemRawPatches =
    kMaximumVisionItemTokens * kRawPatchesPerVisionToken;

namespace frontend {
struct ProcessorOptions {
    std::uint64_t image_min_pixels = 32ULL * 32ULL;
    std::uint64_t image_max_pixels = 1024ULL * 1024ULL;
    std::uint64_t video_min_pixels = 128ULL * 32ULL * 32ULL;
    std::uint64_t video_max_pixels = 4ULL * 1024ULL * 1024ULL;
    // Encoded bytes are aggregate per prompt. Decode limits are per item; the fixed worker pool
    // bounds concurrently decoded media.
    std::size_t max_encoded_media_bytes    = kMaximumPromptMediaBytes;
    std::uint64_t max_decoded_pixels       = 64ULL * 1024ULL * 1024ULL;
    std::uint64_t max_decoded_video_pixels = 128ULL * 1024ULL * 1024ULL;
    int max_video_source_frames            = 100'000;
    double max_video_duration_seconds      = 600.0;
    std::uint64_t max_raw_patches          = kMaximumPromptVisionRawPatches;
    std::uint64_t max_vision_tokens        = kMaximumPromptVisionTokens;
    double video_fps                       = 2.0;
    int video_min_frames                   = 4;
    int video_max_frames                   = 768;
};

} // namespace frontend

struct FrontendResources {
    models::frontend::TextResources text;
    std::string_view preprocessor_config_json;
    std::string_view video_preprocessor_config_json;
    std::optional<frontend::ProcessorOptions> processor;
};

struct MediaGeometry {
    std::uint32_t patch_size;
    std::uint32_t temporal_patch_size;
    std::uint32_t spatial_merge_size;
};

frontend::ProcessorOptions parse_processor_options(std::string_view image_json,
                                                   std::string_view video_json);

void parse_resources(FrontendResources& resources, std::uint32_t embedding_rows,
                     std::optional<MediaGeometry> vision);

} // namespace ninfer::models::qwen
