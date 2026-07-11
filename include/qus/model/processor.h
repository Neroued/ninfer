#pragma once

#include "qus/text/chat_template.h"
#include "qus/text/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace qus::model {

enum class Modality : std::uint8_t {
    Image = 1,
    Video = 2,
};

struct VisionGrid {
    int t = 0;
    int h = 0;
    int w = 0;
};

struct TokenSpan {
    std::size_t begin = 0;
    std::size_t count = 0;
};

struct VisionItem {
    Modality modality = Modality::Image;
    VisionGrid grid;
    std::size_t patch_begin = 0;
    std::size_t patch_count = 0;
    std::vector<double> timestamps;
    std::vector<TokenSpan> token_spans;
};

struct PreprocessStats {
    std::size_t media_items       = 0;
    std::uint64_t raw_patches     = 0;
    std::uint64_t vision_tokens   = 0;
    std::uint64_t attention_pairs = 0;
    std::size_t prompt_tokens     = 0;
    std::size_t patch_bytes       = 0;

    [[nodiscard]] std::string summary() const;
};

struct ProcessorOptions {
    std::uint64_t image_min_pixels         = 32ULL * 32ULL;
    std::uint64_t image_max_pixels         = 1024ULL * 1024ULL;
    std::uint64_t video_min_pixels         = 128ULL * 32ULL * 32ULL;
    std::uint64_t video_max_pixels         = 4ULL * 1024ULL * 1024ULL;
    std::size_t max_media_bytes            = 256ULL << 20;
    std::uint64_t max_decoded_pixels       = 64ULL * 1024ULL * 1024ULL;
    std::uint64_t max_decoded_video_pixels = 128ULL * 1024ULL * 1024ULL;
    int max_video_source_frames            = 100'000;
    double max_video_duration_seconds      = 600.0;
    std::size_t max_media_items            = 16;
    std::uint64_t max_raw_patches          = 131'072;
    std::uint64_t max_vision_tokens        = 32'768;
    std::uint64_t max_attention_pairs      = 128ULL * 1024ULL * 1024ULL;
    std::size_t max_prompt_tokens          = 32'768;
    double video_fps                       = 2.0;
    int video_min_frames                   = 4;
    int video_max_frames                   = 768;
    int fetch_connect_timeout_ms           = 5'000;
    int fetch_timeout_ms                   = 60'000;
    int fetch_max_redirects                = 3;
    bool allow_remote                      = true;
    bool allow_private_network             = false;
    std::filesystem::path media_root;
};

struct ProcessedInput {
    std::vector<int> input_ids;
    std::vector<std::uint8_t> token_types;
    // Axis-major [3, input_ids.size()] in temporal, height, width order.
    std::vector<std::int32_t> positions;
    std::int32_t rope_delta = 0;
    // Row-major [sum(raw_patches), 1536], in the exact merger-friendly order.
    std::vector<float> patches;
    std::vector<VisionItem> vision_items;
    PreprocessStats stats;

    [[nodiscard]] std::span<const std::int32_t> position_axis(int axis) const;
};

class Processor {
public:
    explicit Processor(const text::QwenTokenizer& tokenizer, ProcessorOptions options = {});

    ProcessedInput process(const std::vector<text::ChatMessage>& messages,
                           text::ChatRenderOptions render_options = {}) const;

private:
    const text::QwenTokenizer& tokenizer_;
    ProcessorOptions options_;
};

} // namespace qus::model
